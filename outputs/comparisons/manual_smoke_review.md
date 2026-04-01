# Manual Smoke Review

Smoke-run prediction artifacts were not found. This review covers all 20 smoke samples and records a blocker row for each expected model/prompt pair because raw model outputs are unavailable.

| sample_id | text | gold_intent | gold_entities | slice_tags | sample-level note | gold concern |
| --- | --- | --- | --- | --- | --- | --- |
| raw_012 | Monday ko 11 baje Rahul ko call set kar | call_request | datetime:Monday ko 11 baje; contact:Rahul | temporal_reference, prompt_language_hinglish | Gold looks clean and defensible. |  |
| raw_018 | aaj ka weather Gurgaon ka check karo | information_query | place:Gurgaon; date:aaj | temporal_reference, prompt_language_hinglish | Gold looks clean and defensible. |  |
| raw_026 | mtg ko wed 3 bje reschd krdo | meeting_schedule | event_name:mtg; datetime:wed 3 bje | transliteration_noise, temporal_reference, prompt_language_hinglish | Gold looks acceptable. |  |
| raw_031 | 2 ghnte baad pani pine ka yaad dila | reminder_create | duration:2 ghnte; task_object:pani pina | transliteration_noise, temporal_reference, prompt_language_hinglish | Gold looks acceptable. |  |
| raw_035 | kabir ke liye bday gift buy krna add kr | purchase_request | person:kabir; task_object:bday gift | transliteration_noise, prompt_language_hinglish | Could be read as purchase_request or task_create because of add kr; keep if label policy prefers underlying user task over task-list wrapper. | intent may be borderline between purchase_request and task_create |
| raw_036 | pls vol kum krdo | media_control | other:volume | transliteration_noise, short_utterance, prompt_language_hinglish | Gold label is fine; entity type other is broad but defensible. |  |
| raw_038 | aaj ki notes me likh 'wifi fir gya' | note_create | date:aaj; task_object:wifi fir gya | transliteration_noise, prompt_language_hinglish | Gold looks acceptable; transliteration noise is useful. |  |
| raw_046 | add task: courier pickup before 5 warna miss ho jayega | task_create | task_object:courier pickup; time:before 5 | code_switching, temporal_reference, prompt_language_hinglish | Gold looks acceptable. |  |
| raw_048 | calendar me block kar do deep work 2 to 4 pm | meeting_schedule | event_name:deep work; datetime:2 to 4 pm | code_switching, temporal_reference, prompt_language_hinglish | Gold looks acceptable. |  |
| raw_051 | book a cab, mujhe airport by 6:30 chahiye | navigation_request | destination:airport; time:6:30 | code_switching, temporal_reference, prompt_language_hinglish | Gold looks acceptable. |  |
| raw_062 | kal wala reminder thoda late kar do | reminder_update | date:kal | ambiguity, temporal_reference, prompt_language_hinglish | Ambiguous but still defensible for reminder_update. |  |
| raw_074 | kal ka plan cancel hai ya shift, check kar | information_query | date:kal; event_name:plan | ambiguity, temporal_reference, prompt_language_hinglish | Information query label is defensible. |  |
| raw_077 | weekend pe ek call fix kar dena chachu ke saath | call_request | contact:chachu; date:weekend | ambiguity, temporal_reference, prompt_language_hinglish | Ambiguous timing but intent is still defendable. |  |
| raw_079 | abhi nahi, thodi der me location bhejna | message_send | duration:thodi der me; message_content:location | ambiguity, temporal_reference, prompt_language_hinglish | Gold looks acceptable. |  |
| raw_081 | don't remind me to not skip gym kal | reminder_create | task_object:gym skip na karna; date:kal | adversarial, code_switching, temporal_reference, prompt_language_hinglish | Adversarial negation is intentional and useful. |  |
| raw_083 | set alarm 5 ka... nahi 5:30... wait weekdays only | reminder_update | time:5:30; date:weekdays | ambiguity, adversarial, temporal_reference, prompt_language_hinglish | Adversarial correction pattern is useful and gold label is defensible. |  |
| raw_085 | meeting cancel mat karna, bas push to next monday | meeting_schedule | date:next monday; event_name:meeting | adversarial, temporal_reference, prompt_language_hinglish | Gold looks acceptable. |  |
| raw_094 | call Aman for 5 mins, stretch to 30 if he cries | call_request | contact:Aman; duration:5 mins; duration:30 | adversarial, sentiment_load, prompt_language_hinglish | Second duration 30 is conditional and may create misleading entity misses; sample is useful but gold policy should explicitly allow conditional alternatives. | conditional duration entity may be unstable for extraction scoring |
| raw_096 | schedule standup at 9 IST and 9 PST dono | meeting_schedule | time:9 IST; time:9 PST; event_name:standup | ambiguity, adversarial, code_switching, prompt_language_hinglish | Intent is fine, but dual times without explicit relation may trigger noisy entity or failure interpretations; acceptable only if adversarial ambiguity is intentional. | sample is intentionally hard and underspecified |
| raw_100 | delete reminder called 'pay rent' but keep rent reminder | reminder_cancel | task_object:pay rent | ambiguity, adversarial, prompt_language_hinglish | Gold intent is fine, but the distinction between pay rent and rent reminder is annotation-sensitive; keep only if naming-reference ambiguity is desired. | entity surface may be too annotation-sensitive |

## Expected Model/Prompt Pairs

- `sarvam-30b` / `hinglish`
- `sarvam-30b` / `english`
- `gpt-4o-mini` / `hinglish`
- `gpt-4o-mini` / `english`

## Review Status

- Reviewed rows recorded in CSV: 80
- Prediction files found: 0
- Per-sample evaluation files found: 0
- Manual output inspection status: blocked until a real smoke run is executed
