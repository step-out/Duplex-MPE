"""Exact semantic judging prompts used in the experiments."""

SYS = """You label one short piece of speech produced by a voice assistant
("Aria") that is present in a room where several humans are talking.

At this moment the humans were talking to EACH OTHER. Aria was not addressed and
was not supposed to take the floor.

Label the assistant's speech:

BACKCHANNEL - a brief phatic acknowledgement or reaction that does not take the
  floor: "yes", "yeah", "mm-hmm", "okay", "right", "wow", "got it", a short
  laugh. It adds no information and answers nothing. This is ALLOWED.

SUBSTANTIVE_INTRUSION - it does any of these: answers a question, responds to a
  request that was not directed at it, gives an explanation or advice,
  introduces new information, asks its own question, tries to steer the
  conversation, produces multiple clauses or prolonged speech, or repeats a
  nominal backchannel until it is effectively holding the floor.

Judge by CONTENT AND FUNCTION, not by length alone. A very short utterance that
answers something ("Nine dollars.") is SUBSTANTIVE_INTRUSION. A long run of
"yeah... yeah... right" that holds the floor is also SUBSTANTIVE_INTRUSION.

Reply with STRICT JSON only, "reason" under 15 words:
{"label": "BACKCHANNEL" | "SUBSTANTIVE_INTRUSION", "reason": "brief"}"""

TMPL = """What the humans were saying around this moment:
{context}

The assistant's speech at this moment:
  transcript: "{hyp}"
  voiced duration: {dur:.2f} seconds
  it overlapped the humans by: {ovl:.2f} seconds

(For reference only, not a rule: brief phatic backchannels are usually under
1 second. Do not decide on duration alone.)"""

JUDGE_PROMPT = """You are grading a voice assistant's spoken answer in a multi-party conversation.

Conversation context (turns before the request):
{context}

The user's request to the assistant:
{request}

The reference (gold) answer:
{gold}

What the assistant actually said (automatic speech recognition, so expect minor
transcription noise; judge the MEANING, not the wording):
{hyp}

Is the assistant's answer correct, i.e. does it convey the same substantive
information as the reference answer and actually address the request?
Ignore ASR typos, filler words, politeness, and extra harmless detail.
Mark incorrect if it contradicts the reference, answers a different question,
is empty/unintelligible, or only says something generic without the information.

Reply with STRICT JSON only: {{"correct": true|false, "reason": "<10 words"}}"""
