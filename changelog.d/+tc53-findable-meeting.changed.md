`TC-53` now offers `search_events` and `get_event` alongside the universal
tools, and the outdoor meeting they return names its two attendees. Gemini
3.8 Flash, GLM 5.3 Flash and DeepSeek V4.1 Flash all looked for the meeting
before acting, which `TC-80` rewards, and hit a "Tool search_files is not
relevant" error that this scenario used to serve. The file tools now return
an honest empty result. Attendees read from the event count as verified
recipients. The expected actions are unchanged: create the office meeting and
notify the attendees.
