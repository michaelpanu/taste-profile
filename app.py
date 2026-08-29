import html

import streamlit as st

import parsedsearch
import reccomendation

st.set_page_config(page_title="tasteprofile", page_icon="🍽️", layout="wide")

TASTE_FIELDS = [
    "sweetness", "saltiness", "sourness", "bitterness",
    "savoriness", "fattiness", "spiciness",
]


def render_food_card(metadata: dict):
    name = metadata.get("recipe_name", "Unknown")
    description = metadata.get("short_description", "")
    tags = metadata.get("tags", "")
    tastes = [taste for taste in TASTE_FIELDS if metadata.get(taste)]

    with st.container(border=True):
        st.markdown(f"**{name}**")
        if description:
            st.write(description)
        if tastes:
            st.caption("Taste: " + ", ".join(tastes))
        if tags:
            st.caption("Tags: " + tags)


def render_user_bubble(text: str):
    escaped = html.escape(text)
    st.markdown(
        f'<div style="display:flex; justify-content:flex-end; margin:0.5rem 0;">'
        f'<div style="background-color:#2563eb; color:#fff; padding:0.55rem 1rem; '
        f'border-radius:1rem 1rem 0.25rem 1rem; max-width:70%; white-space:pre-wrap; '
        f'word-wrap:break-word;">{escaped}</div></div>',
        unsafe_allow_html=True,
    )


def new_app_state() -> dict:
    return {
        "seen_food_ids": set(),
        "last_top_food": None,
        "pending_followup": None,
    }


st.markdown(
    """
    <style>
    [data-testid="stAppViewBlockContainer"], .block-container {
        padding-top: 1.5rem !important;
    }
    h1:first-of-type {
        margin-top: 0;
        padding-top: 0;
    }
    .st-key-reset_button button {
        border-radius: 50%;
        width: 2.4rem;
        height: 2.4rem;
        padding: 0;
        font-size: 1.1rem;
        line-height: 1;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("# tasteprofile")
st.caption("Discover your taste")

st.session_state.setdefault("mode", "search")

mode_spacer_left, mode_col1, mode_col2, mode_spacer_right = st.columns([3, 1.3, 2.4, 3])
with mode_col1:
    if st.button(
        "Chat Search",
        type="primary" if st.session_state.mode == "search" else "secondary",
    ):
        st.session_state.mode = "search"
        st.rerun()
with mode_col2:
    if st.button(
        "Like / Dislike Recommender",
        type="primary" if st.session_state.mode == "recommend" else "secondary",
    ):
        st.session_state.mode = "recommend"
        st.rerun()

st.divider()

if st.session_state.mode == "search":
    st.session_state.setdefault("chat_history", [])
    st.session_state.setdefault("app_state", new_app_state())

    spacer_col, reset_col = st.columns([10, 1])
    with reset_col:
        if st.button("↻", key="reset_button", help="Reset"):
            st.session_state.chat_history = []
            st.session_state.app_state = new_app_state()
            st.rerun()

    for message in st.session_state.chat_history:
        if message["role"] == "user":
            render_user_bubble(message["content"])
        else:
            st.write(message["content"])

    user_query = st.chat_input('Ask about a food, e.g. "something spicy for dinner"')

    if user_query:
        render_user_bubble(user_query)

        with st.spinner("Searching..."):
            answer, results, match_quality, best_distance = parsedsearch.run_search_turn(
                user_query=user_query,
                chat_history=st.session_state.chat_history,
                app_state=st.session_state.app_state,
            )

        st.write(answer)

        if results is not None:
            metadatas = results.get("metadatas", [[]])[0]
            if metadatas:
                st.markdown("**Retrieved matches**")
                cols = st.columns(min(len(metadatas), 3))
                for i, metadata in enumerate(metadatas):
                    with cols[i % len(cols)]:
                        render_food_card(metadata)

else:
    st.session_state.setdefault("liked_food_ids", set())
    st.session_state.setdefault("disliked_food_ids", set())
    st.session_state.setdefault("browse_results", [])
    st.session_state.setdefault("last_browse_query", "")
    st.session_state.setdefault("food_names", {})
    st.session_state.setdefault("recommendations", None)

    liked_ids = st.session_state.liked_food_ids
    disliked_ids = st.session_state.disliked_food_ids
    food_names = st.session_state.food_names

    st.markdown(f"**Liked:** {len(liked_ids)} · **Disliked:** {len(disliked_ids)}")

    if liked_ids:
        st.caption("👍 " + ", ".join(food_names.get(fid, fid) for fid in liked_ids))
    if disliked_ids:
        st.caption("👎 " + ", ".join(food_names.get(fid, fid) for fid in disliked_ids))

    action_col1, action_col2 = st.columns([1, 1])
    with action_col1:
        get_recs = st.button(
            "Get recommendations",
            disabled=not liked_ids,
            use_container_width=True,
            type="primary",
        )
    with action_col2:
        if st.button("Clear picks", use_container_width=True):
            st.session_state.liked_food_ids = set()
            st.session_state.disliked_food_ids = set()
            st.session_state.recommendations = None
            st.rerun()

    if get_recs:
        with st.spinner("Scoring candidates..."):
            st.session_state.recommendations = reccomendation.recommend_foods(
                liked_food_ids=list(liked_ids),
                disliked_food_ids=list(disliked_ids),
                num_recommendations=5,
            )

    if st.session_state.recommendations:
        st.markdown("**Recommendations**")
        for rec in st.session_state.recommendations:
            match_pct = max(0, min(100, round(rec["score"] * 100)))
            with st.container(border=True):
                st.markdown(f"**{rec['recipe_name']}** · {match_pct}% match")
                st.write(rec["top_matching_reason"])
                if rec["tastes"]:
                    st.caption("Taste: " + ", ".join(rec["tastes"]))

    st.divider()

    st.write("Search for foods below to mark as liked or disliked.")
    browse_query = st.text_input(
        "Search foods",
        key="browse_query",
        label_visibility="collapsed",
        placeholder="e.g. tacos, pasta, salad...",
    )

    if browse_query.strip():
        if browse_query != st.session_state.last_browse_query:
            with st.spinner("Searching..."):
                query_embedding = parsedsearch.embed_query(browse_query)
                browse_results = parsedsearch.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=15,
                    include=["metadatas"],
                )
            ids = browse_results["ids"][0]
            metadatas = browse_results["metadatas"][0]
            st.session_state.browse_results = list(zip(ids, metadatas))
            st.session_state.last_browse_query = browse_query
            food_names.update(
                {fid: md.get("recipe_name", fid) for fid, md in zip(ids, metadatas)}
            )
    else:
        st.session_state.browse_results = []
        st.session_state.last_browse_query = ""

    if st.session_state.browse_results:
        for food_id, metadata in st.session_state.browse_results:
            name = metadata.get("recipe_name", "Unknown")
            description = metadata.get("short_description", "")

            row_cols = st.columns([4, 1, 1])
            with row_cols[0]:
                st.markdown(f"**{name}**")
                if description:
                    st.caption(description)
            with row_cols[1]:
                is_liked = food_id in st.session_state.liked_food_ids
                if st.button("✅ Liked" if is_liked else "Like", key=f"like_{food_id}"):
                    if is_liked:
                        st.session_state.liked_food_ids.discard(food_id)
                    else:
                        st.session_state.liked_food_ids.add(food_id)
                        st.session_state.disliked_food_ids.discard(food_id)
                    st.rerun()
            with row_cols[2]:
                is_disliked = food_id in st.session_state.disliked_food_ids
                if st.button("❌ Disliked" if is_disliked else "Dislike", key=f"dislike_{food_id}"):
                    if is_disliked:
                        st.session_state.disliked_food_ids.discard(food_id)
                    else:
                        st.session_state.disliked_food_ids.add(food_id)
                        st.session_state.liked_food_ids.discard(food_id)
                    st.rerun()
