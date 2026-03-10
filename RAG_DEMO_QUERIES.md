# 🎯 Demo RAG Query Examples - Practice These!

## 📝 **BEST RAG QUERIES TO DEMONSTRATE** (During Live Demo)

These queries showcase Pathway's live RAG capabilities perfectly. Practice typing these smoothly!

---

## 🔴 **SCENARIO 1: High-Risk Patient Analysis**

**Setup:** Click on Patient P003 (should be yellow/red risk)

### Query 1: Basic Risk Assessment
```
Why is Patient P003 high-risk?
```

**Expected AI Response Type:**
- Specific vital trends with numbers (HR: 85→108, SpO2: 98→91)
- Shock index analysis (current value vs. threshold)
- Clinical interpretation (early septic shock, tissue hypoperfusion)
- Time context ("over the last 2 hours")

**🎤 What to say during demo:**
> "Notice the AI doesn't just say 'high heart rate'—it gives me **specific trends** with actual numbers. This data comes from Pathway's vector store, which indexes the last 3 hours of vitals in real-time."

---

### Query 2: Follow-Up Action Plan
```
What should I do for Patient P003?
```

**Expected AI Response Type:**
- Clinical recommendations (check lactate, blood cultures, consider antibiotics)
- Monitoring priorities (watch BP, oxygen saturation)
- Escalation criteria (when to call rapid response)

**🎤 What to say:**
> "This is clinical decision support powered by Pathway's RAG. The AI retrieves relevant vital patterns and provides contextual recommendations."

---

### Query 3: Time-Specific Query (Tests RAG Freshness)
```
What happened to Patient P003 in the last 30 minutes?
```

**Expected AI Response Type:**
- Recent vital changes (within last 30 min)
- Trend direction (improving/deteriorating)
- Specific events (if any spikes occurred)

**🎤 What to say:**
> "This query tests Pathway's time filtering. The AI searches only recent documents—embeddings are tagged with timestamps, and Pathway's metadata filtering ensures we get the right time window."

---

## 🟢 **SCENARIO 2: Stable Patient (Baseline Comparison)**

**Setup:** Click on Patient P001 or P002 (should be green/stable)

### Query 4: Baseline Status Check
```
How is Patient P001 doing?
```

**Expected AI Response Type:**
- "Patient P001 is currently stable"
- Vitals within normal ranges
- No active anomalies

**🎤 What to say:**
> "For stable patients, the AI confirms normal status. This reduces alarm fatigue—clinicians aren't overwhelmed with alerts for patients who are doing fine."

---

## 🔥 **SCENARIO 3: Crisis Trigger (WOW MOMENT)**

**Setup:** Open Developer Tools → Select Patient P007 → Click "Trigger Sepsis Spike"

### Query 5: Immediate Crisis Detection
```
What just happened to Patient P007?
```

**Expected AI Response Type:**
- Describes the sudden vital changes (HR spike, BP drop, SpO2 fall)
- Notes the rapid deterioration
- Flags the crisis nature

**🎤 What to say (KILLER LINE):**
> "I just triggered a sepsis crisis 10 seconds ago. Watch—Pathway has ALREADY indexed this event. The AI sees it immediately because:
> 1. New vitals hit Kafka
> 2. Pathway chunks them into text
> 3. Embeddings generate incrementally
> 4. Vector store updates (no batch reindex)
> 5. RAG query retrieves the fresh data
> 
> Total time: **1-2 seconds**. This is live streaming RAG in action."

---

### Query 6: Immediate Action Query
```
What should I do RIGHT NOW for Patient P007?
```

**Expected AI Response Type:**
- Urgent recommendations (activate rapid response, notify physician immediately)
- Critical interventions (oxygen support, fluid resuscitation)
- Monitoring escalation

**🎤 What to say:**
> "The AI prioritizes urgency. For a crisis patient, it gives immediate action steps—not just observations. This is where Pathway's real-time capability saves minutes in clinical response."

---

## 🔍 **SCENARIO 4: Comparative Analysis**

### Query 7: Multi-Patient Comparison
```
Compare Patient P003 and Patient P005
```

**Expected AI Response Type:**
- Risk level comparison
- Key vital differences
- Triage priority recommendation

**🎤 What to say:**
> "Pathway's metadata filtering lets us query multiple patients. The AI retrieves vitals from both, compares trends, and helps prioritize who needs attention first."

---

### Query 8: Floor-Level Overview
```
Which patients need attention right now?
```

**Expected AI Response Type:**
- List of high/medium risk patients
- Brief summary of each patient's status
- Priority ranking

**🎤 What to say:**
> "In a real ICU, nurses handle 4-8 patients. This query helps them triage—who's critical, who's stable, who's trending worse. Pathway indexes ALL patients' vitals simultaneously."

---

## 📊 **SCENARIO 5: Vital-Specific Deep Dive**

### Query 9: Single Vital Trend Analysis
```
Why is Patient P003's oxygen saturation dropping?
```

**Expected AI Response Type:**
- SpO2 trend over time (specific values)
- Correlation with other vitals (HR up, RR up)
- Possible clinical causes (sepsis, pneumonia, pulmonary issues)

**🎤 What to say:**
> "This shows semantic search in action. I asked about 'oxygen saturation'—the AI retrieves chunks mentioning 'SpO2', 'hypoxia', 'desaturation'. Pathway's embeddings understand clinical synonyms."

---

### Query 10: Shock Index Explanation
```
Explain the shock index for Patient P003
```

**Expected AI Response Type:**
- Shock index definition (HR / SBP)
- Current value vs. normal range (<0.7 normal, >1.0 critical)
- Clinical significance (tissue perfusion indicator)

**🎤 What to say:**
> "This tests the AI's ability to retrieve calculated features. Shock index is computed by Pathway's feature engineering pipeline—the RAG service sees the enriched vitals, not just raw data."

---

## 🧪 **SCENARIO 6: Edge Cases (If Time Allows)**

### Query 11: Negative Query (No Alert)
```
Are there any patients in critical danger?
```

**Expected AI Response Type:**
- If none: "All patients are currently stable" or list only high-risk ones
- If some: List critical patients with brief summaries

**🎤 What to say:**
> "This tests Pathway's filtering. If no patients meet 'critical danger' criteria, the AI won't hallucinate problems—it accurately reports the current state."

---

### Query 12: Historical Trend Query
```
How has Patient P003's condition changed over the last hour?
```

**Expected AI Response Type:**
- Time-series summary (start vs. end vitals)
- Trend direction (improving/stable/deteriorating)
- Key inflection points (when did HR spike?)

**🎤 What to say:**
> "Pathway's vector store retains 3 hours of vital history. The AI can see patterns: 'Patient was stable until 45 minutes ago, then HR started climbing.' This temporal awareness is crucial for clinical context."

---

## 🎯 **DEMO BEST PRACTICES**

### **DO:**
- ✅ Type queries naturally (not copy-paste) — shows confidence
- ✅ Read AI responses out loud — highlight specific numbers/trends
- ✅ Point to dashboard while explaining — visual connection
- ✅ Use the crisis trigger (Query 5) — it's your WOW moment
- ✅ Mention "Pathway" explicitly when explaining RAG freshness

### **DON'T:**
- ❌ Rush through queries — let the AI response load, then explain
- ❌ Query patients you haven't shown on dashboard — context matters
- ❌ Ask queries if RAG service is down — check endpoints first
- ❌ Ignore AI hallucinations — if response is wrong, acknowledge and move on
- ❌ Skip the follow-up query — shows conversational capability

---

## 🔧 **TECHNICAL DETAILS TO MENTION** (While Waiting for AI Response)

While the AI is "thinking" (1-2 seconds), fill the silence with:

**During Query 1-4 (Normal queries):**
> "Behind the scenes, Pathway is:
> 1. Converting my query to a 384-dimensional embedding
> 2. Searching the vector store (KNN with k=8)
> 3. Retrieving the 8 most similar vital chunks
> 4. Passing them to Groq's LLaMA model
> 5. Generating a clinical response"

**During Query 5-6 (Crisis queries):**
> "This is impressive because:
> - The crisis happened 15 seconds ago
> - Pathway has already embedded the new vitals
> - The vector store is updated (no batch reindex)
> - The AI sees it immediately
> 
> In traditional RAG systems, you'd need to:
> 1. Stop the service
> 2. Re-ingest documents
> 3. Rebuild the entire vector index
> 4. Restart the service
> 
> Pathway does this **continuously, incrementally, in real-time**."

---

## 🚨 **IF RAG QUERY FAILS** (Backup Plan)

**Symptom:** Query returns error or timeout

**Recovery script:**
> "Technical difficulty with the LLM API—but this shows the resilience of our architecture. Notice the dashboard is still updating—that's because the Pathway feature engineering pipeline (Pipeline 1) is independent. Only the RAG service (Pipeline 2) is affected. In production, we'd failover to a backup LLM endpoint. Let me show you the actual Pathway code instead..."

[Then switch to showing code in pathway_pipeline.py]

---

## 📈 **QUERY PROGRESSION FOR STORYTELLING**

Use this sequence to build narrative tension:

1. **Query 4** (stable patient) — "Everything's fine here"
2. **Query 1** (high-risk patient) — "But this patient worries me"
3. **Query 3** (time-specific) — "What changed recently?"
4. **Query 2** (action plan) — "What should I do about it?"
5. **TRIGGER CRISIS** (Developer Tools) — "Let's see how the system responds to emergencies"
6. **Query 5** (crisis detection) — "BOOM—already detected in 2 seconds"
7. **Query 6** (urgent action) — "And here's what to do RIGHT NOW"

This builds from calm → concern → crisis → resolution. It tells a story.

---

## 🎬 **PRACTICE SCRIPT** (Rehearse This!)

**[Click Patient P003]**

**You:** "Patient P003 is showing yellow/red risk. Let me ask the AI..."

**[Type Query 1]** `Why is Patient P003 high-risk?`

**[While waiting]:** "Pathway is searching 3 hours of vital history, finding relevant chunks..."

**[Response appears]:** "Look at these specific trends—HR from 85 to 108, SpO2 from 98 to 91. This isn't a static report; these numbers are from the **live stream**."

**[Type Query 2]** `What should I do for Patient P003?`

**[Response appears]:** "Clinical decision support—check lactate, consider antibiotics. This context-aware advice comes from Pathway's RAG."

**[Open Developer Tools]**

**You:** "Now, let's test real-time updates. I'm triggering a crisis for Patient P007..."

**[Click Trigger Sepsis Spike]**

**[Wait 5-10 seconds, watch dashboard change colors]**

**You:** "See that? Patient went from green to red. Let's ask the AI..."

**[Type Query 5]** `What just happened to Patient P007?`

**[Response appears]:** "IT KNOWS! This event happened **10 seconds ago**, and the AI already sees it. That's because Pathway updates embeddings **incrementally**—no batch reindexing. This is live streaming RAG."

**[Close strong]:** "In an ICU, every second matters. Pathway's real-time AI gives clinicians superhuman pattern recognition."

---

## ✅ **PRE-DEMO CHECKLIST**

Before starting:
- [ ] All Docker containers running (`docker ps` shows 8 services)
- [ ] Dashboard loads (http://localhost:3000)
- [ ] At least 1 patient is yellow/red (if all green, wait 1-2 minutes)
- [ ] RAG endpoint responding (`curl http://localhost:8000/api/handoff/query -X POST -H "Content-Type: application/json" -d '{"query":"test","patient_id":"P001"}'`)
- [ ] Developer Tools button visible on dashboard
- [ ] Browser zoom at 100% (for visibility)
- [ ] Tabs open: Dashboard, Code (pathway_pipeline.py), Architecture diagram

Good luck! 🚀
