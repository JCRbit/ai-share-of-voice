from collections import defaultdict

import pytest
from pydantic import ValidationError

from ai_share_of_voice.agents.specs import (
    AGENT_SPECS,
    KNOWN_TOOLS,
    MAX_TOOLS_PER_AGENT,
    SHARED_TOOLS,
    WEB_TOOLS,
    AgentSpec,
)

EXPECTED_AGENTS = [
    "question_generator",
    "mention_extractor",
    "analyst",
    "source_researcher",
    "visual_analyst",
    "skeptic",
    "writer",
]


def make_spec(**overrides: object) -> AgentSpec:
    data: dict[str, object] = {
        "key": "test_agent",
        "role": "Test Agent",
        "goal": "Check that the spec validation accepts a reasonable goal.",
        "backstory": "You are a test agent used only to exercise the spec validation rules.",
        "tools": ("stats_tool",),
        "max_iter": 3,
    }
    data.update(overrides)
    return AgentSpec(**data)


def tool_users() -> dict[str, list[str]]:
    users: dict[str, list[str]] = defaultdict(list)
    for key, spec in AGENT_SPECS.items():
        for tool in spec.tools:
            users[tool].append(key)
    return users


# ---- the crew ---------------------------------------------------------------------------


def test_the_crew_has_the_seven_agents_in_flow_order() -> None:
    assert list(AGENT_SPECS) == EXPECTED_AGENTS


def test_every_agent_has_a_role_goal_and_backstory() -> None:
    for key, spec in AGENT_SPECS.items():
        assert spec.key == key
        assert spec.role and spec.goal and spec.backstory


def test_roles_do_not_overlap() -> None:
    roles = [spec.role.casefold() for spec in AGENT_SPECS.values()]
    assert len(roles) == len(set(roles))


def test_no_agent_can_delegate() -> None:
    assert all(spec.allow_delegation is False for spec in AGENT_SPECS.values())


def test_iteration_limits_keep_costs_bounded() -> None:
    assert all(spec.max_iter <= 8 for spec in AGENT_SPECS.values())


def test_every_agent_has_at_most_three_tools() -> None:
    assert all(len(spec.tools) <= MAX_TOOLS_PER_AGENT for spec in AGENT_SPECS.values())


def test_texts_have_no_braces_that_crewai_would_treat_as_placeholders() -> None:
    for spec in AGENT_SPECS.values():
        text = spec.role + spec.goal + spec.backstory
        assert "{" not in text
        assert "}" not in text


# ---- tools ------------------------------------------------------------------------------


def test_every_tool_of_every_agent_is_known() -> None:
    for spec in AGENT_SPECS.values():
        assert set(spec.tools) <= set(KNOWN_TOOLS)


def test_every_known_tool_is_used_by_some_agent() -> None:
    assert set(tool_users()) == set(KNOWN_TOOLS)


def test_a_tool_is_shared_only_when_it_is_read_only() -> None:
    shared = {tool for tool, users in tool_users().items() if len(users) > 1}
    assert shared <= SHARED_TOOLS


def test_only_the_source_researcher_reaches_the_web() -> None:
    for tool in WEB_TOOLS:
        assert tool_users()[tool] == ["source_researcher"]


def test_the_skeptic_has_no_web_access() -> None:
    assert not set(AGENT_SPECS["skeptic"].tools) & WEB_TOOLS


def test_only_the_analyst_uses_the_statistics_tool() -> None:
    assert tool_users()["stats_tool"] == ["analyst"]


def test_the_question_generator_needs_no_tools() -> None:
    assert AGENT_SPECS["question_generator"].tools == ()


# ---- spec validation --------------------------------------------------------------------


def test_a_valid_spec_is_accepted_and_immutable() -> None:
    spec = make_spec()
    assert spec.allow_delegation is False
    with pytest.raises(ValidationError):
        spec.role = "Another role"  # type: ignore[misc]


@pytest.mark.parametrize(
    "overrides",
    [
        {"tools": ("does_not_exist",)},
        {"tools": ("stats_tool", "stats_tool")},
        {"tools": tuple(KNOWN_TOOLS)},
        {"goal": "A goal with a {placeholder} inside that CrewAI would interpolate."},
        {"backstory": "A backstory with {braces} that CrewAI would try to fill in later."},
        {"allow_delegation": True},
        {"key": "Not-A-Slug"},
        {"max_iter": 0},
        {"max_iter": 16},
        {"role": "ab"},
        {"goal": "too short"},
        {"unexpected": "value"},
    ],
    ids=[
        "unknown_tool",
        "duplicate_tool",
        "too_many_tools",
        "placeholder_in_goal",
        "placeholder_in_backstory",
        "delegation_enabled",
        "bad_key",
        "zero_iterations",
        "too_many_iterations",
        "short_role",
        "short_goal",
        "unknown_field",
    ],
)
def test_invalid_specs_are_rejected(overrides: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        make_spec(**overrides)
