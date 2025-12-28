import json
import os
from typing import Any, Dict, Tuple

from dotenv import load_dotenv
import streamlit as st
import google.generativeai as genai

# -----------------------------
# Configuration
# -----------------------------

MODEL_NAME = "gemini-2.5-flash"

@st.cache_resource
def get_model():
    # Load environment variables from .env (if present)
    load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is missing. Set it in your environment or .env file."
        )
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(MODEL_NAME)


# -----------------------------
# JSON schema + validation
# -----------------------------

REQUIRED_KEYS = {
    "role": str,
    "summary": str,
    "key_findings": str,
    "confidence_or_risk_score": (int, float),
    "recommendation": str,
    "reason": str,
}

VALID_ROLES = {"planner", "executor", "validator"}
VALID_RECOMMENDATIONS = {"continue", "approve", "block"}


def validate_role_json(payload: Dict[str, Any]) -> Tuple[bool, str]:
    """Validate that the model output matches the governance JSON schema."""
    if not isinstance(payload, dict):
        return False, "Output is not a JSON object"

    for key, expected_type in REQUIRED_KEYS.items():
        if key not in payload:
            return False, f"Missing key: {key}"
        if not isinstance(payload[key], expected_type):
            return False, f"Key '{key}' must be of type {expected_type}, got {type(payload[key])}"

    if payload["role"] not in VALID_ROLES:
        return False, f"Invalid role '{payload['role']}', expected one of {VALID_ROLES}"

    if payload["recommendation"] not in VALID_RECOMMENDATIONS:
        return False, (
            f"Invalid recommendation '{payload['recommendation']}', "
            f"expected one of {VALID_RECOMMENDATIONS}"
        )

    score = float(payload["confidence_or_risk_score"])
    if not (0.0 <= score <= 1.0):
        return False, "confidence_or_risk_score must be between 0 and 1"

    return True, "OK"


# -----------------------------
# Role prompting
# -----------------------------

BASE_SYSTEM_RULES = """
SYSTEM RULES (MANDATORY):
1. You must always respect system-defined rules and policies.
2. You are NOT the final decision-maker. Final decisions are taken by the system.
3. You must explain your reasoning clearly and concisely.
4. You must NEVER bypass validation, rules, or approval requirements.
5. If information is missing or risky, you must flag it instead of guessing.
6. You must always return structured output in JSON.

You must return JSON in this exact schema and nothing else:

{
  "role": "<planner | executor | validator>",
  "summary": "...",
  "key_findings": "...",
  "confidence_or_risk_score": 0.xx,
  "recommendation": "<continue | approve | block>",
  "reason": "clear explanation"
}

Return ONLY valid JSON. Do not include markdown, comments, or extra text.
""".strip()


def build_prompt_for_role(
    role: str,
    user_payload: Dict[str, Any],
) -> str:
    """
    Build a governance-aware prompt for a given role.
    user_payload is arbitrary and can include task, rules, prior outputs, etc.
    """
    # Role-specific instructions
    if role == "planner":
        role_instructions = """
ROLE: PLANNER

- Understand the user task and any rules or policies.
- Break it into required steps and sub-tasks.
- Identify what validations or checks are needed.
- Do NOT generate final content for the user.
- Focus on structure, dependencies, and risk points.
- If information is missing, clearly flag what is needed.

Your "recommendation" field should usually be "continue" unless the task
is clearly impossible or clearly violates given rules.
""".strip()
    elif role == "executor":
        role_instructions = """
ROLE: EXECUTOR

- Take the planner's structure and the original task.
- Generate a DRAFT answer or solution based on the plan.
- Extract any key values, decisions, or outputs into the "key_findings" field.
- Provide an overall numeric confidence score between 0 and 1
  in "confidence_or_risk_score".
- Treat this as a draft: you are NOT the final authority.
- If there are major uncertainties or missing information, clearly state this.

Your "recommendation" field should usually be "continue" to send the draft
to the validator. Only use "block" if the draft itself would clearly violate
the rules or is too unsafe to present.
""".strip()
    else:  # validator
        role_instructions = """
ROLE: VALIDATOR

- Carefully review the original task, any rules/policies, the planner output,
  and the executor's draft.
- Check for policy or rule violations, safety issues, or missing information.
- Identify specific risks and reference them in "key_findings".
- Use "confidence_or_risk_score" as a risk measure:
  - Closer to 1 means HIGHER risk / concern.
  - Closer to 0 means LOWER risk / concern.
- Decide whether to "approve" (safe enough), "block" (too risky or violates rules),
  or "continue" (needs more clarification / another iteration).

Be conservative: if unsure, lean toward "continue" or "block" with a clear reason.
""".strip()

    prompt = f"""
You are part of an AI Decision Control System with multiple roles.

{BASE_SYSTEM_RULES}

{role_instructions}

User payload (JSON):
{json.dumps(user_payload, ensure_ascii=False, indent=2)}
""".strip()

    return prompt


def call_role_model(role: str, user_payload: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
    """
    Call Gemini for a specific role with retry-on-invalid-JSON.
    Returns (parsed_json, raw_text).
    Raises RuntimeError if it cannot obtain valid JSON after retries.
    """
    model = get_model()
    prompt = build_prompt_for_role(role, user_payload)

    last_raw = ""
    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        response = model.generate_content(prompt)
        raw_text = response.text.strip()
        last_raw = raw_text

        # Try to parse JSON
        try:
            parsed = json.loads(raw_text)
        except json.JSONDecodeError:
            # Ask the model to fix JSON
            prompt = (
                f"{prompt}\n\n"
                "Your previous response was not valid JSON. "
                "Return ONLY valid JSON matching the schema, with no extra text."
            )
            continue

        is_valid, msg = validate_role_json(parsed)
        if is_valid:
            return parsed, raw_text

        # If schema invalid, try again with feedback
        prompt = (
            f"{prompt}\n\n"
            f"Your previous response did not match the required schema ({msg}). "
            "Return ONLY valid JSON matching the schema, with no extra text."
        )

    raise RuntimeError(
        f"Failed to obtain valid JSON for role '{role}' after {max_attempts} attempts. "
        f"Last raw response was:\n{last_raw}"
    )


# -----------------------------
# Streamlit UI
# -----------------------------

st.set_page_config(
    page_title="AI Decision Control",
    page_icon="✅",
    layout="wide",
)

st.title("AI Decision Control")

st.markdown(
    """
This app helps you **control and review AI-generated answers** before you use them.

It solves these problems:

- **Unstructured AI outputs** – it first plans how to approach your task.
- **Blind trust in a single answer** – it generates a draft, then checks it again.
- **Lack of risk awareness** – it rates risk/uncertainty and explains why.
- **No human-in-the-loop** – it keeps the final decision in your hands.

You can send **any kind of task** (coding, writing, analysis, comparisons, etc.).
"""
)

with st.expander("Configuration", expanded=False):
    # Check if an AI backend key is available via environment / .env
    load_dotenv()
    api_key_present = bool(os.getenv("GEMINI_API_KEY"))
    st.write(f"AI backend key configured: `{api_key_present}`")

user_task = st.text_area(
    "Task / request",
    placeholder="Describe what you want the system to help decide or produce...",
    height=200,
)

run_button = st.button("Run governed workflow")

if run_button:
    if not user_task.strip():
        st.error("Please enter a task or request first.")
        st.stop()

    # Shared payload for all roles (no explicit rules box; validation is automatic)
    base_payload = {
        "task": user_task.strip(),
    }

    # -------------------------
    # 1. Planner
    # -------------------------
    st.subheader("1. Planner")
    with st.spinner("Planner is analyzing the task and designing steps..."):
        try:
            planner_output, planner_raw = call_role_model("planner", base_payload)
        except Exception as e:
            st.error(f"Planner failed: {e}")
            st.stop()

    st.markdown("**Planner JSON output:**")
    st.json(planner_output)

    # -------------------------
    # 2. Executor
    # -------------------------
    st.subheader("2. Executor")
    executor_payload = {
        **base_payload,
        "planner_output": planner_output,
    }

    with st.spinner("Executor is generating a draft based on the plan..."):
        try:
            executor_output, executor_raw = call_role_model("executor", executor_payload)
        except Exception as e:
            st.error(f"Executor failed: {e}")
            st.stop()

    st.markdown("**Executor JSON output:**")
    st.json(executor_output)

    # -------------------------
    # 3. Validator
    # -------------------------
    st.subheader("3. Validator")
    validator_payload = {
        **base_payload,
        "planner_output": planner_output,
        "executor_output": executor_output,
    }

    with st.spinner("Validator is checking the draft against rules and risks..."):
        try:
            validator_output, validator_raw = call_role_model("validator", validator_payload)
        except Exception as e:
            st.error(f"Validator failed: {e}")
            st.stop()

    st.markdown("**Validator JSON output:**")
    st.json(validator_output)

    # -------------------------
    # Final decision (human in the loop)
    # -------------------------
    st.markdown("---")
    st.subheader("Final decision (human-controlled)")

    rec = validator_output.get("recommendation", "continue")
    risk_score = float(validator_output.get("confidence_or_risk_score", 0.0))
    reason = validator_output.get("reason", "")

    col_dec1, col_dec2 = st.columns(2)

    with col_dec1:
        st.markdown("**Validator recommendation:**")
        st.write(f"Recommendation: `{rec}`")
        st.write(f"Risk/Confidence score: `{risk_score:.2f}`")

    with col_dec2:
        st.markdown("**Validator reasoning:**")
        st.write(reason or "(No reason provided)")

    st.info(
        "The validator is **not** the final authority. "
        "Use your judgment before acting on this recommendation."
    )

    final_choice = st.radio(
        "Your final decision:",
        options=["Undecided", "Approve", "Block", "Request Changes"],
        index=0,
    )

    if final_choice != "Undecided":
        st.success(f"You selected final decision: **{final_choice}**")

