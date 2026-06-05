import os
import json
import time
from pathlib import Path
from typing import Any

from google import genai


class GeminiExpander:
    def __init__(
        self,
        model_name="gemini-3.1-flash-lite-preview",
        cache_dir="./cache/gemini_expansions",
        sleep_seconds=4.5,
        max_retries=5,
    ):
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.sleep_seconds = sleep_seconds
        self.max_retries = max_retries

        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is not set.")

        self.client = genai.Client(api_key=api_key)

    def build_prompt(self, conversation_text: str) -> str:
        return f"""
Given the conversation below, generate 5 reference songs that represent the user's current music request.

These reference songs are NOT final recommendations. They are only used as metadata-style query expansion for embedding-based retrieval.

Return only valid JSON. Do not use markdown. Do not add explanations.

The output MUST be a JSON list of exactly 5 objects.
Each object MUST follow this schema:
{{
  "track_name": [string],
  "artist_name": [string],
  "album_name": [string],
  "tag_list": [list of genre, mood, theme, instrumentation, energy, and style tags],
  "release_date": string
}}

Rules:
- Assume the current turn is the last user turn in the conversation.
- Focus on the current user request.
- Use previous turns only to understand preferences, dislikes, corrections, and constraints.
- If the user asks for new artists, avoid artists already recommended or repeatedly mentioned.
- track_name, artist_name, and album_name MUST be lists with one string inside.
- tag_list MUST be a list of strings.
- release_date MUST be a string.
- Do not include track_id, ISRC, artist_id, album_id, popularity, or duration.
- Focus on musical attributes that would help retrieve similar tracks from a catalog.

Conversation:
{conversation_text}
"""

    def _to_list_of_strings(self, value: Any, default: str = "") -> list[str]:
        """
        Convert a value into a clean list of strings.

        This helper normalizes Gemini fields that may be returned as strings,
        lists, numbers, or missing values. Empty values are replaced with the
        provided default when available.

        Args:
            value: Value to normalize.
            default: Fallback string used when the value is empty or missing.

        Returns:
            A list of non-empty strings.
        """
        if value is None:
            return [default] if default else []

        if isinstance(value, list):
            cleaned = [str(v).strip() for v in value if str(v).strip()]
            return cleaned if cleaned else ([default] if default else [])

        if isinstance(value, str):
            value = value.strip()
            return [value] if value else ([default] if default else [])

        return [str(value).strip()] if str(value).strip() else ([default] if default else [])

    def _strip_markdown_json(self, text: str) -> str:
        """
        Remove markdown code fences from a Gemini JSON response.

        Gemini may sometimes wrap JSON in markdown blocks such as ```json ... ```.
        This helper removes those wrappers and returns only the raw JSON text.

        Args:
            text: Raw response text from Gemini.

        Returns:
            Response text with markdown JSON fences removed.
        """
        text = text.strip()

        if text.startswith("```json"):
            text = text[len("```json"):].strip()
            if text.endswith("```"):
                text = text[:-3].strip()
        elif text.startswith("```"):
            text = text[len("```"):].strip()
            if text.endswith("```"):
                text = text[:-3].strip()

        return text.strip()

    def _extract_json_text(self, text: str) -> str:
        """
        Extra safety: if Gemini adds text before/after JSON, extract the JSON part.
        """
        text = self._strip_markdown_json(text)

        if text.startswith("[") and text.endswith("]"):
            return text

        if text.startswith("{") and text.endswith("}"):
            return text

        start_list = text.find("[")
        end_list = text.rfind("]")

        if start_list != -1 and end_list != -1 and end_list > start_list:
            return text[start_list:end_list + 1]

        start_obj = text.find("{")
        end_obj = text.rfind("}")

        if start_obj != -1 and end_obj != -1 and end_obj > start_obj:
            return text[start_obj:end_obj + 1]

        raise ValueError("Could not find JSON in Gemini response.")

    def _normalize_pseudo_tracks(self, pseudo_tracks: Any) -> list[dict[str, Any]]:
        """
        Converts Gemini output into a reliable list of track dictionaries.
        Handles:
        - list[dict]
        - {"songs": list[dict]}
        - {"tracks": list[dict]}
        - list[str]
        """
        if isinstance(pseudo_tracks, dict):
            for key in ["songs", "tracks", "recommendations", "reference_songs", "reference_tracks", "items"]:
                if key in pseudo_tracks and isinstance(pseudo_tracks[key], list):
                    pseudo_tracks = pseudo_tracks[key]
                    break

        if not isinstance(pseudo_tracks, list):
            raise ValueError("Gemini output is not a list.")

        cleaned_tracks = []

        for i, track in enumerate(pseudo_tracks):
            if isinstance(track, str):
                # Salvage string-only output instead of crashing.
                cleaned_tracks.append({
                    "track_name": [track],
                    "artist_name": ["Unknown Artist"],
                    "album_name": ["Unknown Album"],
                    "tag_list": ["music recommendation", "similar style"],
                    "release_date": "",
                })
                continue

            if not isinstance(track, dict):
                print(f"Skipping invalid pseudo-track at index {i}: {track}", flush=True)
                continue

            cleaned_track = {
                "track_name": self._to_list_of_strings(track.get("track_name"), "Unknown Track")[:1],
                "artist_name": self._to_list_of_strings(track.get("artist_name"), "Unknown Artist")[:1],
                "album_name": self._to_list_of_strings(track.get("album_name"), "Unknown Album")[:1],
                "tag_list": self._to_list_of_strings(track.get("tag_list"), "music recommendation"),
                "release_date": str(track.get("release_date", "")).strip(),
            }

            cleaned_tracks.append(cleaned_track)

        if not cleaned_tracks:
            raise ValueError("No valid pseudo-tracks found in Gemini output.")

        return cleaned_tracks[:5]

    def pseudo_tracks_to_query2(self, pseudo_tracks):
        """
        Convert pseudo-track metadata into a text query for retrieval.

        The input is first normalized into a list of track dictionaries. Each track
        is then converted into a multi-line metadata string containing track name,
        artist name, album name, tags, and release date.

        Args:
            pseudo_tracks: Gemini-generated pseudo-track data.

        Returns:
            A string used as the expanded retrieval query.
        """
        pseudo_tracks = self._normalize_pseudo_tracks(pseudo_tracks)

        parts = []

        for track in pseudo_tracks:
            track_name = ", ".join(track["track_name"])
            artist_name = ", ".join(track["artist_name"])
            album_name = ", ".join(track["album_name"])
            tag_list = ", ".join(track["tag_list"])
            release_date = track["release_date"]

            parts.append(
                f"track_name: {track_name}\n"
                f"artist_name: {artist_name}\n"
                f"album_name: {album_name}\n"
                f"tag_list: {tag_list}\n"
                f"release_date: {release_date}"
            )

        return "\n\n".join(parts)
    
    def pseudo_tracks_to_query(self, pseudo_tracks):
        pseudo_tracks = self._normalize_pseudo_tracks(pseudo_tracks)

        all_tags = []

        for track in pseudo_tracks:
            tags = track.get("tag_list", [])
            if isinstance(tags, str):
                tags = [tags]

            all_tags.extend([str(tag).strip() for tag in tags if str(tag).strip()])

        all_tags = list(dict.fromkeys(all_tags))

        if not all_tags:
            return ""

        return "expanded_tags: " + ", ".join(all_tags)

    def _cache_path(self, session_id, turn_number):
        safe_session = str(session_id).replace("/", "_").replace("\\", "_")
        return self.cache_dir / f"{safe_session}_turn_{turn_number}.json"

    def _call_gemini_with_retry(self, prompt: str):
        """
        Call the Gemini API with retry logic for temporary failures.

        This function sends the prompt to the configured Gemini model and retries
        when the API returns temporary errors such as high demand, rate limits,
        or internal server errors.

        Args:
            prompt: The prompt text sent to Gemini.

        Returns:
            The Gemini API response object.

        Raises:
            Exception: Re-raises the final API error if all retry attempts fail,
            or immediately raises non-retryable errors.
        """
        last_error = None

        for attempt in range(1, self.max_retries + 1):
            try:
                print(
                    f"Calling Gemini API... attempt {attempt}/{self.max_retries} using {self.model_name}",
                    flush=True,
                )

                return self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                )

            except Exception as e:
                last_error = e
                error_text = str(e)

                retryable = (
                    "503" in error_text
                    or "UNAVAILABLE" in error_text
                    or "429" in error_text
                    or "RESOURCE_EXHAUSTED" in error_text
                    or "500" in error_text
                    or "INTERNAL" in error_text
                )

                if retryable and attempt < self.max_retries:
                    wait_time = self.sleep_seconds * attempt
                    print(
                        f"Gemini temporary error. Waiting {wait_time:.1f}s before retry. Error: {e}",
                        flush=True,
                    )
                    time.sleep(wait_time)
                    continue

                raise

        raise RuntimeError(f"Gemini failed after {self.max_retries} attempts: {last_error}")

    def expand_tracks(self, conversation_text, session_id=None, turn_number=None):
        """
        Generate or load Gemini reference tracks for a conversation turn.

        If a cached Gemini response exists for the given session and turn, it is
        loaded and normalized. Otherwise, the conversation is
        sent to Gemini, the response is normalized into pseudo-track metadata,
        and saved to cache.

        Args:
            conversation_text: Conversation history and current user request.
            session_id: Optional session identifier used for caching.
            turn_number: Optional turn number used for caching.

        Returns:
            A normalized list of up to five Gemini reference track dictionaries.
        """
        cache_path = None

        if session_id is not None and turn_number is not None:
            cache_path = self._cache_path(session_id, turn_number)

            if cache_path.exists():
                try:
                    print(f"Using cached Gemini expansion: {cache_path}", flush=True)
                    with open(cache_path, "r", encoding="utf-8") as f:
                        pseudo_tracks = json.load(f)

                    return self._normalize_pseudo_tracks(pseudo_tracks)

                except Exception as e:
                    print(f"Cached Gemini expansion invalid. Deleting cache file. Error: {e}", flush=True)
                    try:
                        cache_path.unlink()
                    except OSError:
                        pass

        prompt = self.build_prompt(conversation_text)

        response = self._call_gemini_with_retry(prompt)

        if not response.text:
            raise ValueError("Gemini returned an empty response.")

        raw_text = response.text.strip()

        try:
            json_text = self._extract_json_text(raw_text)
            pseudo_tracks = json.loads(json_text)
            pseudo_tracks = self._normalize_pseudo_tracks(pseudo_tracks)
        except Exception as e:
            print("Gemini returned invalid or unexpected JSON.", flush=True)
            print("Raw Gemini response:", flush=True)
            print(raw_text, flush=True)
            raise e

        if cache_path is not None:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(pseudo_tracks, f, ensure_ascii=False, indent=2)

        time.sleep(self.sleep_seconds)

        return pseudo_tracks

    def expand(self, conversation_text, session_id=None, turn_number=None):
        """
        Generate or load a Gemini-based query expansion for a conversation turn.

        Returns:
            A metadata-style query string built from Gemini pseudo-tracks.
        """
        pseudo_tracks = self.expand_tracks(
            conversation_text,
            session_id=session_id,
            turn_number=turn_number,
        )
        return self.pseudo_tracks_to_query(pseudo_tracks)
