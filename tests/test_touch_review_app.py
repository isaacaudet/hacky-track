import json
import tempfile
import unittest
from pathlib import Path

from touch_review_app import HTML, TouchReviewStore


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


class TouchReviewAppTests(unittest.TestCase):
    def make_store(self, root: Path, items: list[dict] | None = None) -> TouchReviewStore:
        manifest = root / "touch_review_manifest.json"
        candidates = root / "audio_candidates"
        labels = root / "visual_touch_labels"
        write_json(
            manifest,
            {
                "items": items
                or [
                    {
                        "video_id": "video-a",
                        "video_name": "video-a.MOV",
                        "video_path": str(root / "video-a.MOV"),
                        "split": "train",
                    }
                ]
            },
        )
        candidates.mkdir(parents=True, exist_ok=True)
        return TouchReviewStore(manifest, candidates, labels)

    def write_candidates(self, root: Path, video_id: str, times: list[float]) -> None:
        write_json(
            root / "audio_candidates" / f"{video_id}.touch_candidates.json",
            {
                "audio_candidates": [{"time_sec": time_sec, "strength": 1.0} for time_sec in times],
                "existing_event_hints": [],
            },
        )

    def test_draft_and_complete_saves_are_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = self.make_store(root)
            self.write_candidates(root, "video-a", [1.2, 2.0])

            draft = store.save_events(
                "video-a",
                [{"type": "touch", "time_sec": 1.2345, "review_status": "approved"}],
                candidate_review_complete=False,
                candidate_reviews=[{"time_sec": 1.2, "decision": "touch"}],
            )
            self.assertFalse(draft["candidate_review_complete"])
            self.assertEqual(draft["review_status"], "in_progress")
            self.assertIsNone(draft["review_completed_at"])
            self.assertEqual(draft["candidate_reviews"][0]["decision"], "touch")

            state = store.state()
            self.assertEqual(state["summary"]["complete_videos"], 0)

            complete = store.save_events(
                "video-a",
                [{"type": "touch", "time_sec": 1.2345, "review_status": "approved"}],
                candidate_review_complete=True,
                candidate_reviews=[
                    {"time_sec": 1.2, "decision": "touch"},
                    {"time_sec": 2.0, "decision": "no_touch"},
                ],
            )
            self.assertTrue(complete["candidate_review_complete"])
            self.assertEqual(complete["review_status"], "complete")
            self.assertIsNotNone(complete["review_completed_at"])
            self.assertEqual(len(complete["candidate_reviews"]), 2)
            self.assertEqual(store.state()["summary"]["complete_videos"], 1)
            self.assertEqual(store.state()["summary"]["reviewed_hints"], 2)

    def test_save_preserves_v1_contact_labels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = self.make_store(root)

            draft = store.save_events(
                "video-a",
                [
                    {
                        "type": "touch",
                        "time_sec": 1.2345,
                        "review_status": "approved",
                        "contact_side": "right",
                        "contact_type": "kick",
                        "contact_surface": "outer",
                        "trick_label": "right_outer_kick",
                        "contact_review_status": "reviewed",
                        "contact_side_basis": "wearer_limb",
                    }
                ],
                candidate_review_complete=False,
                candidate_reviews=[],
            )
            event = draft["rallies"][0]["events"][0]

            self.assertEqual(event["contact_side"], "right")
            self.assertEqual(event["contact_type"], "kick")
            self.assertEqual(event["contact_surface"], "outer")
            self.assertEqual(event["trick_label"], "right_outer_kick")
            self.assertEqual(event["contact_review_status"], "reviewed")
            self.assertEqual(event["contact_side_basis"], "wearer_limb")

    def test_save_preserves_generic_side_contact_labels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = self.make_store(root)

            draft = store.save_events(
                "video-a",
                [
                    {
                        "type": "touch",
                        "time_sec": 1.2345,
                        "review_status": "approved",
                        "contact_side": "left",
                        "contact_type": "kick",
                        "contact_surface": "unknown",
                        "trick_label": "left_kick",
                        "contact_review_status": "reviewed",
                    }
                ],
                candidate_review_complete=False,
                candidate_reviews=[],
            )
            event = draft["rallies"][0]["events"][0]

            self.assertEqual(event["contact_side"], "left")
            self.assertEqual(event["contact_type"], "kick")
            self.assertEqual(event["contact_surface"], "unknown")
            self.assertEqual(event["trick_label"], "left_kick")
            self.assertEqual(event["contact_side_basis"], "wearer_limb")

    def test_state_counts_wearer_and_legacy_side_basis_labels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = self.make_store(root)
            store.save_events(
                "video-a",
                [
                    {
                        "type": "touch",
                        "time_sec": 1.0,
                        "review_status": "approved",
                        "contact_side": "left",
                        "contact_side_basis": "wearer_limb",
                        "contact_type": "kick",
                    },
                    {
                        "type": "touch",
                        "time_sec": 2.0,
                        "review_status": "approved",
                        "contact_side": "right",
                        "contact_type": "kick",
                    },
                    {
                        "type": "touch",
                        "time_sec": 3.0,
                        "review_status": "approved",
                        "contact_side": "left",
                        "contact_side_basis": "screen_position",
                        "contact_type": "kick",
                    },
                ],
                candidate_review_complete=False,
                candidate_reviews=[],
            )

            state = store.state()
            item = state["items"][0]

            self.assertEqual(item["wearer_side_events"], 1)
            self.assertEqual(item["legacy_side_basis_events"], 1)
            self.assertEqual(item["ambiguous_side_basis_events"], 1)
            self.assertEqual(state["summary"]["wearer_side_events"], 1)
            self.assertEqual(state["summary"]["legacy_side_basis_events"], 1)

    def test_save_adds_contact_defaults_for_stall_and_drop(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = self.make_store(root)

            draft = store.save_events(
                "video-a",
                [
                    {"type": "stall", "time_sec": 2.0, "review_status": "approved"},
                    {"type": "drop_floor", "time_sec": 3.0, "review_status": "approved"},
                ],
                candidate_review_complete=False,
                candidate_reviews=[],
            )
            events = draft["rallies"][0]["events"]

            self.assertEqual(events[0]["contact_type"], "stall")
            self.assertEqual(events[1]["contact_type"], "ground")
            self.assertEqual(events[0]["contact_surface"], "unknown")
            self.assertEqual(events[1]["contact_surface"], "unknown")
            self.assertEqual(events[0]["contact_review_status"], "unreviewed")
            self.assertEqual(events[1]["contact_review_status"], "unreviewed")

    def test_video_payload_exposes_contact_sheet_links(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = self.make_store(
                root,
                items=[
                    {
                        "video_id": "video-a",
                        "video_name": "video-a.MOV",
                        "video_path": str(root / "video-a.MOV"),
                        "split": "test_frozen",
                    }
                ],
            )
            store.contact_sheets_dir = root / "touch_review_contact_sheets"
            sheet_dir = store.contact_sheets_dir / "video-a"
            sheet_dir.mkdir(parents=True)
            (sheet_dir / "video-a_audio_only_p001.png").write_bytes(b"png")
            (sheet_dir / "video-a_likely_unchecked_p001.png").write_bytes(b"png")
            write_json(
                store.contact_sheets_dir / "touch_review_contact_sheets.json",
                {
                    "videos": [
                        {
                            "video_id": "video-a",
                            "sheets": [
                                {
                                    "path": str(sheet_dir / "video-a_audio_only_p001.png"),
                                    "candidate_count": 3,
                                    "times_sec": [1.2, 1.6, 2.0],
                                },
                                {
                                    "path": str(sheet_dir / "video-a_likely_unchecked_p001.png"),
                                    "candidate_count": 1,
                                    "times_sec": [3.4],
                                },
                            ],
                        }
                    ]
                },
            )
            store.contact_sheet_manifest = store.load_contact_sheet_manifest()

            payload = store.video_payload("video-a")
            sheets = payload["contact_sheets"]

            self.assertEqual([row["filter"] for row in sheets], ["audio_only", "likely_unchecked"])
            self.assertEqual(sheets[0]["label"], "Audio-tail")
            self.assertEqual(sheets[1]["label"], "Likely/model")
            self.assertIn("/contact-sheet?id=video-a&name=", sheets[0]["url"])
            self.assertEqual(sheets[0]["candidate_count"], 3)
            self.assertEqual(sheets[0]["times_sec"], [1.2, 1.6, 2.0])

    def test_video_payload_exposes_montage_links(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = self.make_store(
                root,
                items=[
                    {
                        "video_id": "video-a",
                        "video_name": "video-a.MOV",
                        "video_path": str(root / "video-a.MOV"),
                        "split": "test_frozen",
                    }
                ],
            )
            store.montages_dir = root / "touch_review_montages"
            montage_dir = store.montages_dir / "video-a"
            montage_dir.mkdir(parents=True)
            (montage_dir / "video-a_audio_only_p001.mp4").write_bytes(b"mp4")
            write_json(
                store.montages_dir / "touch_review_montages.json",
                {
                    "videos": [
                        {
                            "video_id": "video-a",
                            "montages": [
                                {
                                    "path": str(montage_dir / "video-a_audio_only_p001.mp4"),
                                    "candidate_count": 3,
                                    "duration_sec": 2.1,
                                    "times_sec": [1.2, 1.6, 2.0],
                                }
                            ],
                        }
                    ]
                },
            )
            store.montage_manifest = store.load_montage_manifest()

            payload = store.video_payload("video-a")
            montages = payload["review_montages"]

            self.assertEqual(len(montages), 1)
            self.assertEqual(montages[0]["filter"], "audio_only")
            self.assertEqual(montages[0]["label"], "Audio-tail")
            self.assertIn("/review-montage?id=video-a&name=", montages[0]["url"])
            self.assertEqual(montages[0]["candidate_count"], 3)
            self.assertEqual(montages[0]["duration_sec"], 2.1)

    def test_video_payload_exposes_detector_track_graph(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            track_root = root / "tracks"
            write_json(
                track_root / "video-a" / "detector_track.json",
                {
                    "track": [
                        {"frame_index": 1, "time_sec": 0.1, "center": [10, 20], "confidence": 0.8, "source": "model_detection"},
                        {"frame_index": 2, "time_sec": 0.2, "center": [11, 25], "confidence": 0.7, "source": "model_detection"},
                    ]
                },
            )
            manifest = root / "touch_review_manifest.json"
            candidates = root / "audio_candidates"
            labels = root / "visual_touch_labels"
            write_json(
                manifest,
                {
                    "items": [
                        {
                            "video_id": "video-a",
                            "video_name": "video-a.MOV",
                            "video_path": str(root / "video-a.MOV"),
                            "split": "train",
                        }
                    ]
                },
            )
            candidates.mkdir(parents=True, exist_ok=True)
            store = TouchReviewStore(manifest, candidates, labels, track_roots=[track_root])

            graph = store.video_payload("video-a")["track_graph"]

            self.assertEqual(graph["status"], "ok")
            self.assertEqual(len(graph["track_points"]), 2)
            self.assertEqual(graph["track_points"][1]["y"], 25.0)

    def test_complete_save_requires_all_candidate_hints_reviewed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = self.make_store(root)
            self.write_candidates(root, "video-a", [1.2, 2.0])

            with self.assertRaisesRegex(ValueError, "cannot complete: 1 unchecked hints"):
                store.save_events(
                    "video-a",
                    [{"type": "touch", "time_sec": 1.2, "review_status": "approved"}],
                    candidate_review_complete=True,
                    candidate_reviews=[{"time_sec": 1.2, "decision": "touch"}],
                )

            self.assertFalse((root / "visual_touch_labels" / "video-a.events.json").exists())

    def test_complete_save_counts_clustered_candidate_moments(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = self.make_store(root)
            self.write_candidates(root, "video-a", [1.2, 1.22, 2.0])

            complete = store.save_events(
                "video-a",
                [{"type": "touch", "time_sec": 1.22, "review_status": "approved"}],
                candidate_review_complete=True,
                candidate_reviews=[
                    {"time_sec": 1.22, "decision": "touch"},
                    {"time_sec": 2.0, "decision": "no_touch"},
                ],
            )

            self.assertTrue(complete["candidate_review_complete"])
            payload = store.video_payload("video-a")
            self.assertEqual(len(payload["candidates"]["audio_candidates"]), 3)
            self.assertEqual(len(payload["candidate_clusters"]), 2)

    def test_complete_save_requires_candidate_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = self.make_store(root)

            with self.assertRaisesRegex(ValueError, "missing candidate file"):
                store.save_events(
                    "video-a",
                    [],
                    candidate_review_complete=True,
                    candidate_reviews=[],
                )

            self.assertFalse((root / "visual_touch_labels" / "video-a.events.json").exists())

    def test_review_ui_does_not_auto_mark_distant_hint_for_touch(self) -> None:
        self.assertIn("const HINT_ACTION_TOL_SEC = 0.20;", HTML)
        self.assertIn("nearestCandidateDistance(candidate) <= HINT_ACTION_TOL_SEC", HTML)
        self.assertIn('setCandidateReviewFor(candidate, "touch")', HTML)
        self.assertNotIn('if(type === "touch") setCandidateReview("touch");', HTML)

    def test_review_ui_completion_requires_explicit_hint_review(self) -> None:
        self.assertIn("function candidateCovered(c)", HTML)
        self.assertIn("return Boolean(reviewForCandidate(c));", HTML)
        self.assertIn("not saved complete: ${missing.length} unchecked hints remain", HTML)
        self.assertIn("candidate_clusters", HTML)

    def test_review_ui_marks_drop_and_stall_hints_as_no_touch(self) -> None:
        self.assertIn('if(type === "drop_floor" || type === "stall")', HTML)
        self.assertIn('setCandidateReviewFor(candidate, "no_touch")', HTML)

    def test_review_ui_has_fast_unchecked_hint_controls(self) -> None:
        self.assertIn('id="markNoTouchNext"', HTML)
        self.assertIn('id="addTouchNext"', HTML)
        self.assertIn('id="workflowGuide"', HTML)
        self.assertIn("function renderWorkflowGuide()", HTML)
        self.assertIn("Current task", HTML)
        self.assertIn("Jump to next unchecked", HTML)
        self.assertIn("Yes, touch → next", HTML)
        self.assertIn("No, not touch → next", HTML)
        self.assertIn("function stepAnyUncheckedCandidate()", HTML)
        self.assertIn("function addTouchForNearestHint(", HTML)
        self.assertIn("function setNearestCandidateReview(", HTML)
        self.assertIn("What Counts", HTML)
        self.assertIn('id="nextLikelyUnchecked"', HTML)
        self.assertIn('id="likelyReadout"', HTML)
        self.assertIn('id="candidateFilter"', HTML)
        self.assertIn('id="bulkNoTouchFiltered"', HTML)
        self.assertIn('id="bulkAudioTailComplete"', HTML)
        self.assertIn('<option value="likely_unchecked">Likely unchecked</option>', HTML)
        self.assertIn("function filteredCandidates()", HTML)
        self.assertIn("function candidateMatchesFilter(c)", HTML)
        self.assertIn("function nearestFilteredCandidate()", HTML)
        self.assertIn("return nearestCandidateFrom(filteredCandidates());", HTML)
        self.assertIn("const candidate = nearestFilteredCandidate();", HTML)
        self.assertIn("function bulkNoTouchFiltered()", HTML)
        self.assertIn("function upsertCandidateReview(", HTML)
        self.assertIn("bulk no-touch is only available with the Audio only filter", HTML)
        self.assertIn("muted_visual_review_bulk", HTML)
        self.assertIn("function bulkAudioTailAndSaveComplete()", HTML)
        self.assertIn("muted_visual_review_audio_tail_complete", HTML)
        self.assertIn("function sheetTimesNoTouch(rawTimes)", HTML)
        self.assertIn("muted_visual_review_sheet_bulk", HTML)
        self.assertIn('data-sheet-no-touch="${sheetTimes.map(time => time.toFixed(3)).join(",")}"', HTML)
        self.assertIn("no-touch page", HTML)
        self.assertIn("no unchecked audio-tail hints left on this sheet", HTML)
        self.assertIn("reviewedTimes.length", HTML)
        self.assertIn('class="sheet-time${stateClass}"', HTML)
        self.assertIn('review.decision === "touch" ? "T" : "N"', HTML)
        self.assertIn(".sheet-time-grid button.touch", HTML)
        self.assertIn("review ${stats.likelyUnchecked} likely/model hints before bulk-completing audio-tail", HTML)
        self.assertIn("await save(true);", HTML)
        self.assertIn('event.key === "A" && event.shiftKey', HTML)
        self.assertIn('event.key === "N"', HTML)
        self.assertIn('id="nextUnchecked"', HTML)
        self.assertIn("stepUncheckedCandidate(1)", HTML)
        self.assertIn("function likelyUncheckedCandidates()", HTML)
        self.assertIn("function stepLikelyUncheckedCandidate()", HTML)
        self.assertIn("all likely hints checked", HTML)
        self.assertIn("no unchecked hints in current filter", HTML)
        self.assertIn('event.key === "l"', HTML)
        self.assertIn('id="replayHint"', HTML)
        self.assertIn('id="autoReplay"', HTML)

    def test_review_ui_has_v1_contact_label_controls(self) -> None:
        self.assertIn('id="selectedContactSide"', HTML)
        self.assertIn('id="selectedContactType"', HTML)
        self.assertIn('id="selectedContactSurface"', HTML)
        self.assertIn('id="selectedTrickLabel"', HTML)
        self.assertIn("function updateSelectedContactField(", HTML)
        self.assertIn("v1.0 contact labels for selected event", HTML)

    def test_review_ui_has_fast_contact_classification_workflow(self) -> None:
        self.assertIn('id="classifyLeftKick"', HTML)
        self.assertIn('id="classifyLeftOuterKick"', HTML)
        self.assertIn('id="classifyLeftInnerKick"', HTML)
        self.assertIn('id="classifyRightKick"', HTML)
        self.assertIn('id="classifyRightInnerKick"', HTML)
        self.assertIn('id="classifyRightOuterKick"', HTML)
        self.assertIn('id="classifyLeftKnee"', HTML)
        self.assertIn('id="classifyLeftOuterKnee"', HTML)
        self.assertIn('id="classifyLeftInnerKnee"', HTML)
        self.assertIn('id="classifyRightKnee"', HTML)
        self.assertIn('id="classifyRightInnerKnee"', HTML)
        self.assertIn('id="classifyRightOuterKnee"', HTML)
        self.assertIn('id="classifyLeftStall"', HTML)
        self.assertIn('id="classifyLeftInnerStall"', HTML)
        self.assertIn('id="classifyLeftOuterStall"', HTML)
        self.assertIn('id="classifyRightStall"', HTML)
        self.assertIn('id="classifyRightInnerStall"', HTML)
        self.assertIn('id="classifyRightOuterStall"', HTML)
        self.assertIn('id="classifyGround"', HTML)
        self.assertIn('id="classifyUnknown"', HTML)
        self.assertIn('id="nextUnclassifiedEvent"', HTML)
        self.assertIn('id="nextSideBasisReview"', HTML)
        self.assertIn('id="selectedContactSideBasis"', HTML)
        self.assertIn("Side means contacting limb / wearer side", HTML)
        self.assertIn('contact_side_basis:"wearer_limb"', HTML)
        self.assertIn("function classifySelectedEvent(", HTML)
        self.assertIn("function stepUnclassifiedEvent()", HTML)
        self.assertIn("function stepSideBasisReview()", HTML)
        self.assertIn("function sideBasisQueue()", HTML)
        self.assertIn("Load next side-basis clip", HTML)
        self.assertIn("legacy side", HTML)
        self.assertIn("Wearer-side basis labels", HTML)
        self.assertIn("contact_review_status = \"reviewed\"", HTML)
        self.assertIn('event.key === "1"', HTML)
        self.assertIn('event.key === "8"', HTML)
        self.assertIn('event.key === "g"', HTML)
        self.assertIn('event.key === "m"', HTML)
        self.assertIn("autoReplay:true", HTML)
        self.assertIn("function cueCandidate(candidate, message)", HTML)
        self.assertIn("function playCandidateWindow(candidate, message)", HTML)
        self.assertIn('id="frameBack"', HTML)
        self.assertIn('id="frameForward"', HTML)
        self.assertIn('id="playbackRate"', HTML)
        self.assertIn('id="reviewProgress"', HTML)
        self.assertIn('id="reviewSheets"', HTML)
        self.assertIn('id="reviewMontages"', HTML)
        self.assertIn("function candidateReviewStats()", HTML)
        self.assertIn("function chooseDefaultCandidateFilter()", HTML)
        self.assertIn('app.candidateFilter = "likely_unchecked";', HTML)
        self.assertIn('app.candidateFilter = "audio_only";', HTML)
        self.assertIn("chooseDefaultCandidateFilter();", HTML)
        self.assertIn("function renderReviewProgress()", HTML)
        self.assertIn("function renderReviewSheets()", HTML)
        self.assertIn("function renderReviewMontages()", HTML)
        self.assertIn("build_touch_review_montages.py", HTML)
        self.assertIn('href="${montage.url}"', HTML)
        self.assertIn("build_touch_review_contact_sheets.py", HTML)
        self.assertIn('href="${sheet.url}"', HTML)
        self.assertIn('data-sheet-jump="${firstTime.toFixed(3)}"', HTML)
        self.assertIn('data-sheet-time="${time.toFixed(3)}"', HTML)
        self.assertIn('class="sheet-time-grid"', HTML)
        self.assertIn('root.querySelectorAll("[data-sheet-jump], [data-sheet-time]")', HTML)
        self.assertIn("sheet jump ${fmt(target)}s", HTML)
        self.assertIn("Likely/model checked", HTML)
        self.assertIn("Audio-only checked", HTML)
        self.assertIn("Save complete", HTML)
        self.assertIn("stats.audioOnlyChecked", HTML)
        self.assertIn("likely left ${item.likely_unchecked_hints}", HTML)
        self.assertIn("audio-tail ${next.audio_only_unchecked_hint_count}/${next.audio_only_hint_count}", HTML)
        self.assertIn('id="trackGraph"', HTML)
        self.assertIn("function renderTrackGraph()", HTML)
        self.assertIn("function localValleys(points)", HTML)
        self.assertIn("valleys = low ball positions", HTML)
        self.assertIn("function stepTrackValley(direction)", HTML)
        self.assertIn('id="nextValley"', HTML)
        self.assertIn('event.key === "v"', HTML)

    def test_review_ui_has_hint_window_playback_helpers(self) -> None:
        self.assertIn("const REVIEW_WINDOW_SEC = 0.35;", HTML)
        self.assertIn("const FRAME_STEP_SEC = 1 / 30;", HTML)
        self.assertIn("function replayHintWindow()", HTML)
        self.assertIn("function frameStep(direction)", HTML)
        self.assertIn("function stopReplayIfNeeded()", HTML)
        self.assertIn("app.replayCenter = center;", HTML)
        self.assertIn("video.currentTime = app.replayCenter === null ? app.replayUntil : app.replayCenter;", HTML)
        self.assertIn("playCandidateWindow(candidate, `replay ${fmt(candidate.time_sec)}s`)", HTML)
        self.assertIn('event.key === "r"', HTML)
        self.assertIn('event.key === "["', HTML)
        self.assertIn('event.key === "]"', HTML)

    def test_review_ui_enforces_muted_playback(self) -> None:
        self.assertIn("function enforceMuted()", HTML)
        self.assertIn("video.volume = 0;", HTML)
        self.assertIn("video.addEventListener(\"volumechange\", enforceMuted)", HTML)
        self.assertIn("video.addEventListener(\"play\", enforceMuted)", HTML)
        self.assertIn("enforceMuted();", HTML)

    def test_review_ui_autosaves_dirty_drafts(self) -> None:
        self.assertIn("function markDirty(", HTML)
        self.assertIn("function scheduleDraftSave()", HTML)
        self.assertIn("function flushDraftSave()", HTML)
        self.assertIn("setTimeout(() => flushDraftSave(), 900)", HTML)
        self.assertIn("candidate_review_complete:false", HTML)
        self.assertIn("await flushDraftSave();", HTML)
        self.assertIn('window.addEventListener("beforeunload"', HTML)

    def test_review_ui_manual_save_waits_for_draft_autosave(self) -> None:
        self.assertIn("if(app.draftSaveInFlight) await app.draftSaveInFlight;", HTML)
        self.assertIn("app.dirty = false;", HTML)
        self.assertIn('markDirty("event deleted")', HTML)

    def test_state_exposes_readiness_queue_with_frozen_test_first(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = self.make_store(
                root,
                items=[
                    {
                        "video_id": "train-a",
                        "video_name": "train-a.MOV",
                        "video_path": str(root / "train-a.MOV"),
                        "split": "train",
                    },
                    {
                        "video_id": "test-a",
                        "video_name": "test-a.MOV",
                        "video_path": str(root / "test-a.MOV"),
                        "split": "test_frozen",
                    },
                ],
            )
            self.write_candidates(root, "train-a", [1.0, 2.0])
            self.write_candidates(root, "test-a", [1.0])

            state = store.state()

            self.assertEqual(state["review_queue"][0]["video_id"], "test-a")
            self.assertEqual(state["review_queue"][0]["priority"], 0)
            by_id = {row["video_id"]: row for row in state["items"]}
            self.assertEqual(by_id["test-a"]["readiness_status"], "missing_label_file")
            self.assertEqual(by_id["train-a"]["unchecked_hints"], 2)
            self.assertEqual(state["summary"]["unchecked_hints"], 3)

    def test_review_ui_defaults_to_priority_queue_and_supports_deep_link(self) -> None:
        self.assertIn("function initialVideoId()", HTML)
        self.assertIn("app.state.review_queue?.[0]?.video_id", HTML)
        self.assertIn('url.searchParams.set("video_id", videoId)', HTML)
        self.assertIn('id="queueCard"', HTML)
        self.assertIn("Load next priority", HTML)
        self.assertIn("Load next different clip", HTML)
        self.assertIn("Current clip is still top priority until all hints are checked and saved complete.", HTML)

    def test_review_ui_can_save_complete_and_advance_to_next_priority(self) -> None:
        self.assertIn('id="saveCompleteNext"', HTML)
        self.assertIn("save(true, true)", HTML)
        self.assertIn("advanceAfterComplete", HTML)
        self.assertIn("app.state.review_queue || []", HTML)
        self.assertIn("button.disabled = completeBlocked", HTML)
        self.assertIn("not saved complete: ${missing.length} unchecked hints remain", HTML)


if __name__ == "__main__":
    unittest.main()
