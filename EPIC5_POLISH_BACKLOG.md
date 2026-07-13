# Epic 5 Polish Backlog

Tracks handoff-doc wishlist items and what happened to each - built now,
deferred to a specific story, or dropped - so we don't re-litigate the
same decision twice. Updated as each story gets reconciled against the
wishlist (see handoff doc's own closing note: "reconcile against list
above, not yet reviewed for conflicts").

## Decided

- **Welcome splash** - built now (before Story 2). Fully self-contained,
  only needed session state, which Story 1 already provided. See
  src/ui/welcome_splash.py.
- **BYO API key + provider picker** - DROPPED for this submission.
  Found a real bug while scoping it: gemini_client.py and
  openrouter_client.py both bake their API client in at import time
  (module-level singleton), so a sidebar key input would have had zero
  effect without also refactoring both clients to build per-call
  instead of once. That refactor is low-risk in isolation, but not
  worth touching shared LLM client code this close to submission for a
  feature that isn't blocking anything (single free-tier key, no real
  multi-key scenario happening). Current .env-based LLM_PROVIDER setup
  stays as-is. Revisit post-submission if wanted.

## Not yet reconciled (waiting on each story's actual task doc)

These are still just wishlist items until we've checked them against
the real documented scope of whichever story would carry them:

- Emotion result cards, trust-adapted phrasing, mixed-emotion display,
  "How I decided" expander, celebration treatment - expected Story 2,
  needs decision engine output + model scores.
- Correction chips, 👍/👎 feedback (separate csv) - expected Story 2 or
  3, needs results to exist first either way.
- Personal insights (per-field emotion frequency) - expected Story 4.

## Pure UI/UX polish (deferred to the dedicated design pass after Epic 5)

Not functional gaps, not blocking anything - purely visual/layout
refinement, explicitly held for the separate UI/UX-focused chat agreed
on earlier rather than done piecemeal mid-Epic-5:

- Analyze button: make full width instead of its current compact size,
  so it reads as the clear primary action (ChatGPT, post-T3 review).
- Add a bit of vertical spacing between the "Settings" subheader and
  the first checkbox - currently a bit cramped (ChatGPT, post-T3
  review).
- Once T4 fills in the Learning Analytics charts, consider wrapping the
  whole section in `st.expander("📈 Learning Analytics", expanded=False)`
  so a growing page doesn't bury the latest analysis result below a
  wall of charts (ChatGPT, post-T3 review - explicitly flagged as "not
  necessary now, something to revisit after T4").
