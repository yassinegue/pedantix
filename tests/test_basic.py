import json
import tempfile
import unittest

from pedantix_project.dataset import WikiPage
from pedantix_project.dataset import extract_intro_from_wikitext
from pedantix_project.llm_policy import (
    INVALID_GUESS_REWARD,
    REPEATED_GUESS_REWARD,
    build_llm_curriculum,
    extract_guess,
    format_prompt_for_model,
    is_valid_guess,
    make_prompt,
    score_completion,
)
from pedantix_project.model import train_tiny_model
from pedantix_project.rl import solve_with_policy, train_rl_policy
from pedantix_project.simulator import PedantixGame
from pedantix_project.solver import solve_page
from pedantix_project.text import canonical_word


class BasicTest(unittest.TestCase):
    def test_canonical_word_handles_simple_variants(self):
        self.assertEqual(canonical_word("Françaises"), canonical_word("français"))
        self.assertEqual(canonical_word("villes"), canonical_word("ville"))
        self.assertEqual(canonical_word("pays"), "pays")

    def test_game_reveals_title_and_solves(self):
        page = WikiPage("Paris", "Paris est la capitale de la France.")
        game = PedantixGame(page)
        result = game.guess("paris")
        self.assertTrue(result.solved)
        self.assertEqual(result.hit_count, 2)

    def test_extract_intro_from_wikitext(self):
        text = """{{Infobox}}
'''Paris''' est la [[capitale]] de la [[France]].

Elle est une grande ville européenne avec une longue histoire.

== Histoire ==
Le reste ne doit pas être inclus.
"""
        intro = extract_intro_from_wikitext(text)
        self.assertIn("Paris est la capitale de la France.", intro)
        self.assertNotIn("Le reste", intro)

    def test_tiny_solver_can_solve_small_closed_set(self):
        pages = [
            WikiPage("Paris", "Paris est la capitale de la France."),
            WikiPage("Football", "Le football est un sport collectif avec un ballon."),
        ]
        model = train_tiny_model(pages, starter_words=["france", "sport"])
        steps = solve_page(pages[0], pages, model, max_steps=20)
        self.assertTrue(steps[-1].solved)

    def test_rl_policy_plays_without_candidate_filtering(self):
        pages = [
            WikiPage("Paris", "Paris est la capitale de la France."),
            WikiPage("Football", "Le football est un sport collectif avec un ballon."),
        ]
        model = train_tiny_model(pages, starter_words=["france", "sport", "paris", "football"])
        policy = train_rl_policy(pages, model, episodes=80, action_size=20, max_steps=12, seed=1)
        steps = solve_with_policy(pages[0], model, policy, max_steps=20)
        self.assertGreater(len(steps), 0)
        self.assertNotIn("candidates", policy.__dict__)

    def test_llm_reward_scores_one_word_completion(self):
        page = WikiPage("Paris", "Paris est la capitale de la France.")
        model = train_tiny_model([page], starter_words=["paris", "france"])
        reward = score_completion(
            title=page.title,
            intro=page.intro,
            history=[],
            completion="MOT: Paris",
            similarity_model=model,
        )
        self.assertTrue(reward.solved)
        self.assertGreater(reward.reward, 400)

    def test_llm_reward_penalizes_non_solving_guess(self):
        page = WikiPage("Paris", "Paris est la capitale de la France.")
        model = train_tiny_model([page], starter_words=["paris", "france"])
        reward = score_completion(
            title=page.title,
            intro=page.intro,
            history=[],
            completion="MOT: football",
            similarity_model=model,
        )
        self.assertFalse(reward.solved)
        self.assertLess(reward.reward, -20)

    def test_llm_reward_strongly_penalizes_repeats(self):
        page = WikiPage("Paris", "Paris est la capitale de la France.")
        model = train_tiny_model([page], starter_words=["paris", "france"])
        reward = score_completion(
            title=page.title,
            intro=page.intro,
            history=[{"guess": "france", "exact": 1, "semantic": 0, "title": 0, "solved": False}],
            completion="MOT: France",
            similarity_model=model,
        )
        self.assertEqual(reward.reward, REPEATED_GUESS_REWARD)

    def test_llm_reward_rejects_qwen_thinking(self):
        page = WikiPage("Paris", "Paris est la capitale de la France.")
        model = train_tiny_model([page], starter_words=["paris", "france"])
        reward = score_completion(
            title=page.title,
            intro=page.intro,
            history=[],
            completion="MOT: think",
            similarity_model=model,
        )
        self.assertEqual(reward.reward, INVALID_GUESS_REWARD)
        self.assertFalse(is_valid_guess("think"))
        self.assertFalse(is_valid_guess("1693"))

    def test_llm_prompt_does_not_include_hidden_title(self):
        prompt = make_prompt([], max_steps=20)
        self.assertIn("MOT:", prompt)
        self.assertNotIn("Paris", prompt)
        self.assertEqual(extract_guess("MOT: France"), "france")
        self.assertEqual(extract_guess("MOT: françaises"), "francaises")
        self.assertEqual(extract_guess("MOT: cœur"), "cœur")

    def test_qwen_prompt_disables_thinking(self):
        prompt = format_prompt_for_model(make_prompt([], max_steps=20), chat_format="qwen")
        self.assertIn("/no_think", prompt)
        self.assertTrue(prompt.endswith("MOT:"))

    def test_qwen_curriculum_completion_does_not_duplicate_prefix(self):
        page = WikiPage("Paris", "Paris est la capitale de la France.")
        model = train_tiny_model([page], starter_words=["paris", "france", "ville"])
        with tempfile.TemporaryDirectory() as tmp:
            pages_path = f"{tmp}/pages.jsonl"
            out_path = f"{tmp}/curriculum.jsonl"
            with open(pages_path, "w", encoding="utf-8") as handle:
                handle.write(json.dumps({"title": page.title, "intro": page.intro}, ensure_ascii=False) + "\n")
            build_llm_curriculum(
                pages_path,
                out_path,
                model,
                sample_size=1,
                states_per_page=1,
                max_steps=10,
                max_title_words=1,
                action_size=20,
                chat_format="qwen",
                seed=1,
            )
            with open(out_path, encoding="utf-8") as handle:
                row = json.loads(handle.readline())
        self.assertTrue(row["prompt"].endswith("MOT:"))
        self.assertTrue(row["completion"].startswith(" "))
        self.assertFalse(row["completion"].lower().startswith("mot:"))
        self.assertNotIn("MOT: MOT:", row["text"])

    def test_oracle_curriculum_reaches_title_word(self):
        page = WikiPage("Paris", "Paris est la capitale de la France.")
        model = train_tiny_model([page], starter_words=["personne", "ville", "france", "paris"])
        with tempfile.TemporaryDirectory() as tmp:
            pages_path = f"{tmp}/pages.jsonl"
            out_path = f"{tmp}/curriculum.jsonl"
            with open(pages_path, "w", encoding="utf-8") as handle:
                handle.write(json.dumps({"title": page.title, "intro": page.intro}, ensure_ascii=False) + "\n")
            count = build_llm_curriculum(
                pages_path,
                out_path,
                model,
                sample_size=1,
                states_per_page=4,
                max_steps=10,
                max_title_words=1,
                action_size=20,
                chat_format="qwen",
                seed=1,
                trajectory_mode="oracle",
            )
            with open(out_path, encoding="utf-8") as handle:
                rows = [json.loads(line) for line in handle]
        self.assertEqual(count, len(rows))
        self.assertIn(" paris", [row["completion"].lower() for row in rows])


if __name__ == "__main__":
    unittest.main()
