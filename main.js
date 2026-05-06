import { GoogleGenerativeAI } from "@google/generative-ai";

// ═══════════════════════════════════════════════════════════
// INSIGHT ENGINE v2.0 — AI-Powered Strategy Simulator
// 위키의 사고법 데이터를 Gemini AI에 주입하여 실전 전략을 생성
// ═══════════════════════════════════════════════════════════

// ─── Thinking Framework System Prompts ─────────────────────
const SYSTEM_PROMPT = `
당신은 세계 최고의 전략 컨설턴트이자 시스템 사고 전문가입니다.
당신의 역할은 사용자의 문제를 분석하고, 가장 적합한 사고 프레임워크를 선택하여 실전 전략을 생성하는 것입니다.

당신이 보유한 3가지 사고 프레임워크:

═══ 프레임워크 1: 일론 머스크의 제1원칙 + 5단계 알고리즘 ═══
- 핵심: 유추(남들이 하니까)를 거부하고 물리적 기초 사실에서부터 재조립
- 5단계 알고리즘 (반드시 순서대로):
  1. 모든 요구사항에 의문을 제기하라 (누가 만든 규칙인가? 실명을 확인하라)
  2. 불필요한 모든 것을 제거하라 (10% 이상 복원하지 않았다면 덜 삭제한 것)
  3. 단순화하고 최적화하라 (존재해서는 안 될 공정을 최적화하지 마라)
  4. 주기를 가속화하라 (정제된 시스템 위에서만 가속)
  5. 자동화하라 (수동 공정이 완성된 후에)
- 적합한 문제: 비용 절감, 프로세스 혁신, 기술적 한계 돌파, 생산성 향상

═══ 프레임워크 2: 심랑의 비대칭 전략 (사상최강의 데릴사위) ═══
- 핵심: 정보의 비대칭성과 시스템의 구조적 허점을 찌르는 전략
- 3대 전술:
  1. 비대칭 자산 활용: 나만 아는 이질적 지식으로 패러다임 전환
  2. 공포 기반 협상: 상대가 얻을 이익보다 "잃기 두려운 것"을 레버리지로 활용
  3. 시스템적 편입: 상위 권력 시스템의 보호망에 의도적으로 진입하여 방어벽 구축
- 적합한 문제: 협상, 정치적 갈등, 자원이 적은 상태의 경쟁, 위기 탈출

═══ 프레임워크 3: 히라이 다카시의 시스템 사고 (1등의 통찰) ═══
- 핵심: 현상이 아닌 구조를 보고, 인과 관계의 고리를 파악
- 분석 도구:
  1. 인과 고리 분석: 강화 루프(Reinforcing)와 균형 루프(Balancing) 식별
  2. 스톡 앤 플로: 축적되는 것(Stock), 유입(Inflow), 유출(Outflow) 매핑
  3. 레버리지 포인트: 작은 힘으로 전체를 바꿀 수 있는 개입 지점 발견
- 적합한 문제: 조직 문화 개선, 장기적 성장 전략, 복잡한 시스템 이해, 반복되는 문제

모든 응답은 반드시 한국어로 작성하십시오.
`;

const PHASE1_PROMPT = `
사용자가 아래의 문제를 제시했습니다. 다음 작업을 수행하십시오:

1. 3개 프레임워크 중 가장 적합한 1개를 선택하고, 각 프레임워크의 적합도를 0~100 점수로 매기십시오.
2. 선택한 프레임워크를 기반으로, 전략 수립에 필요한 핵심 정보를 수집하기 위한 질문 3~4개를 생성하십시오.
   - 질문은 구체적이고 날카로워야 합니다.
   - 각 질문에는 왜 이 정보가 필요한지 간단한 이유를 포함하십시오.

반드시 아래 JSON 형식으로만 응답하십시오 (마크다운 코드블록 없이 순수 JSON):
{
  "selected_framework": "musk" | "shim" | "hirai",
  "framework_scores": {
    "musk": 점수,
    "shim": 점수,
    "hirai": 점수
  },
  "selection_reason": "이 프레임워크를 선택한 이유 (1~2문장)",
  "questions": [
    {
      "id": 1,
      "question": "질문 내용",
      "reason": "이 질문이 필요한 이유"
    }
  ]
}

사용자의 문제:
`;

const PHASE2_PROMPT = `
사용자의 문제와 수집된 정보를 바탕으로 종합 전략 보고서를 생성하십시오.

보고서에 반드시 포함할 항목:
1. 문제의 핵심 구조 분석 (선택된 프레임워크의 렌즈로)
2. 구체적인 실행 전략 (3~5개 액션 아이템)
3. 레버리지 포인트 (가장 적은 노력으로 최대 효과를 낼 수 있는 지점)
4. 리스크 분석 (주요 위험 요소와 대응 방안)
5. 성공 가능성 점수와 근거

반드시 아래 JSON 형식으로만 응답하십시오 (마크다운 코드블록 없이 순수 JSON):
{
  "scores": {
    "success_probability": 0~100,
    "risk_level": 0~100,
    "leverage_impact": 0~100,
    "execution_difficulty": 0~100
  },
  "structure_analysis": "문제의 핵심 구조를 프레임워크 렌즈로 분석한 내용 (3~5문장)",
  "action_items": [
    {
      "title": "액션 제목",
      "description": "구체적 실행 방법",
      "priority": "high" | "medium" | "low"
    }
  ],
  "leverage_point": {
    "title": "레버리지 포인트 제목",
    "description": "왜 이 지점이 가장 효과적인지 (2~3문장)"
  },
  "risks": [
    {
      "risk": "위험 요소",
      "mitigation": "대응 방안"
    }
  ],
  "conclusion": "최종 전략 요약 (2~3문장)"
}

선택된 프레임워크: {{FRAMEWORK}}

사용자의 원래 문제:
{{PROBLEM}}

수집된 추가 정보:
{{ANSWERS}}
`;

// ─── Framework Metadata ────────────────────────────────────
const FRAMEWORK_META = {
  musk: { icon: "🚀", name: "일론 머스크의 알고리즘", color: "#3b82f6" },
  shim: { icon: "⚔️", name: "심랑의 비대칭 전략", color: "#8b5cf6" },
  hirai: { icon: "🌀", name: "히라이 다카시의 시스템 사고", color: "#06b6d4" }
};

// ─── State ─────────────────────────────────────────────────
let genAI = null;
let model = null;
let state = {
  problem: "",
  selectedFramework: null,
  frameworkScores: {},
  questions: [],
  answers: [],
  selectionReason: ""
};

// ─── DOM References ────────────────────────────────────────
const $ = (sel) => document.querySelector(sel);
const screens = {
  setup: $("#setup-screen"),
  problem: $("#problem-screen"),
  questions: $("#questions-screen"),
  report: $("#report-screen")
};

// ─── Navigation ────────────────────────────────────────────
function showScreen(name) {
  Object.values(screens).forEach(s => s.classList.remove("active"));
  screens[name].classList.add("active");
  // Re-trigger animation
  screens[name].style.animation = 'none';
  screens[name].offsetHeight; // reflow
  screens[name].style.animation = '';
}

function showLoading(msg) {
  $("#loading-message").textContent = msg;
  $("#loading-overlay").style.display = "flex";
}

function hideLoading() {
  $("#loading-overlay").style.display = "none";
}

function setButtonLoading(btn, loading) {
  const textEl = btn.querySelector(".btn-text");
  const loadEl = btn.querySelector(".btn-loading");
  if (textEl) textEl.style.display = loading ? "none" : "inline";
  if (loadEl) loadEl.style.display = loading ? "inline-flex" : "none";
  btn.disabled = loading;
}

// ─── AI Call with Retry (handles 429 rate limits) ──────────
async function callAI(prompt, retries = 3) {
  for (let attempt = 1; attempt <= retries; attempt++) {
    try {
      const result = await model.generateContent(prompt);
      return result.response.text();
    } catch (e) {
      const is429 = e.message && e.message.includes("429");
      if (is429 && attempt < retries) {
        const wait = attempt * 30; // 30s, 60s
        $("#loading-message").textContent = `요청 한도 초과. ${wait}초 후 자동 재시도... (${attempt}/${retries})`;
        $("#loading-overlay").style.display = "flex";
        await new Promise(r => setTimeout(r, wait * 1000));
        continue;
      }
      throw e;
    }
  }
}

// ─── Phase 0: API Key Setup ────────────────────────────────
$("#connect-btn").addEventListener("click", () => {
  const key = $("#api-key-input").value.trim();
  if (!key) { alert("API 키를 입력해 주세요."); return; }
  if (!key.startsWith("AIza")) {
    alert("유효하지 않은 키 형식입니다. Google AI Studio에서 발급받은 키를 입력해 주세요.");
    return;
  }

  // Initialize SDK without making an API call (save quota for real analysis)
  genAI = new GoogleGenerativeAI(key);
  model = genAI.getGenerativeModel({
    model: "gemini-2.0-flash",
    systemInstruction: { parts: [{ text: SYSTEM_PROMPT }] }
  });

  localStorage.setItem("gemini_api_key", key);
  $("#status-dot").classList.add("connected");
  $("#status-text").textContent = "AI ENGINE ONLINE";
  showScreen("problem");
});

// Auto-load saved key (pre-fill only, don't auto-connect)
(function initKey() {
  const savedKey = localStorage.getItem("gemini_api_key");
  if (savedKey) {
    $("#api-key-input").value = savedKey;
  }
})();

// ─── Phase 1: Problem Analysis ─────────────────────────────
$("#analyze-btn").addEventListener("click", async () => {
  const problem = $("#problem-input").value.trim();
  if (!problem) { alert("문제를 입력해 주세요."); return; }
  if (!model) { alert("먼저 API 키를 연결해 주세요."); return; }

  state.problem = problem;
  const btn = $("#analyze-btn");
  setButtonLoading(btn, true);

  try {
    showLoading("AI가 최적 사고법을 분석 중...");
    const text = await callAI(PHASE1_PROMPT + problem);

    // Parse JSON (strip markdown code fences if present)
    const cleaned = text.replace(/```json\n?/g, '').replace(/```\n?/g, '').trim();
    const data = JSON.parse(cleaned);

    state.selectedFramework = data.selected_framework;
    state.frameworkScores = data.framework_scores;
    state.questions = data.questions;
    state.selectionReason = data.selection_reason;

    renderQuestionsScreen();
    showScreen("questions");
  } catch (e) {
    console.error(e);
    const is429 = e.message && e.message.includes("429");
    if (is429) {
      alert("API 무료 요청 한도를 초과했습니다. 1~2분 후 다시 시도해 주세요.");
    } else {
      alert("AI 분석 중 오류가 발생했습니다: " + e.message);
    }
  } finally {
    setButtonLoading(btn, false);
  }
});

// ─── Phase 2: Render AI Questions ──────────────────────────
function renderQuestionsScreen() {
  const fw = FRAMEWORK_META[state.selectedFramework];

  // Badge
  $(".lens-icon").textContent = fw.icon;
  $(".lens-name").textContent = fw.name;
  $(".lens-score").textContent = `적합도 ${state.frameworkScores[state.selectedFramework]}%`;

  // Subtitle
  $("#phase2-subtitle").textContent = state.selectionReason;

  // Questions
  const container = $("#ai-questions-container");
  container.innerHTML = "";

  state.questions.forEach((q, i) => {
    const div = document.createElement("div");
    div.className = "ai-question-item";
    div.innerHTML = `
      <div class="q-number">질문 ${i + 1} / ${state.questions.length}</div>
      <div class="q-text">${q.question}</div>
      <div class="hint" style="margin-top:-0.5rem;margin-bottom:0.75rem;">💡 ${q.reason}</div>
      <textarea class="answer-field" data-qid="${q.id}" placeholder="답변을 입력하십시오..."></textarea>
    `;
    container.appendChild(div);
  });

  // Progress
  $("#q-progress").style.width = "100%";
}

// ─── Phase 3: Generate Strategy Report ─────────────────────
$("#generate-btn").addEventListener("click", async () => {
  const answerFields = document.querySelectorAll(".answer-field");
  const answers = [];
  let allFilled = true;

  answerFields.forEach((field, i) => {
    const val = field.value.trim();
    if (!val) allFilled = false;
    answers.push({ question: state.questions[i].question, answer: val });
  });

  if (!allFilled) {
    alert("모든 질문에 답변해 주세요.");
    return;
  }

  state.answers = answers;
  const btn = $("#generate-btn");
  setButtonLoading(btn, true);
  showLoading("AI가 전략 보고서를 설계하고 있습니다...");

  try {
    const answersText = answers.map((a, i) =>
      `Q${i+1}: ${a.question}\nA${i+1}: ${a.answer}`
    ).join("\n\n");

    const prompt = PHASE2_PROMPT
      .replace("{{FRAMEWORK}}", FRAMEWORK_META[state.selectedFramework].name)
      .replace("{{PROBLEM}}", state.problem)
      .replace("{{ANSWERS}}", answersText);

    const text = await callAI(prompt);
    const cleaned = text.replace(/```json\n?/g, '').replace(/```\n?/g, '').trim();
    const report = JSON.parse(cleaned);

    renderReport(report);
    showScreen("report");
  } catch (e) {
    console.error(e);
    alert("전략 생성 중 오류 발생: " + e.message);
  } finally {
    setButtonLoading(btn, false);
    hideLoading();
  }
});

// ─── Render Report ─────────────────────────────────────────
function renderReport(report) {
  // Score Dashboard
  const dashboard = $("#score-dashboard");
  const scores = report.scores;

  const scoreConfig = [
    { key: "success_probability", label: "성공 가능성", barClass: "bar-green" },
    { key: "leverage_impact", label: "레버리지 임팩트", barClass: "bar-purple" },
    { key: "risk_level", label: "리스크 수준", barClass: "bar-red" },
    { key: "execution_difficulty", label: "실행 난이도", barClass: "bar-amber" }
  ];

  dashboard.innerHTML = scoreConfig.map(sc => {
    const val = scores[sc.key] || 0;
    const colorClass = val >= 70 ? "score-high" : val >= 40 ? "score-mid" : "score-low";
    // For risk and difficulty, invert the color logic
    const isInverse = sc.key === "risk_level" || sc.key === "execution_difficulty";
    const displayColor = isInverse
      ? (val >= 70 ? "score-low" : val >= 40 ? "score-mid" : "score-high")
      : colorClass;

    return `
      <div class="score-card">
        <div class="score-label">${sc.label}</div>
        <div class="score-value ${displayColor}">${val}<span style="font-size:1rem;">%</span></div>
        <div class="score-bar"><div class="score-bar-fill ${sc.barClass}" style="width:${val}%"></div></div>
      </div>
    `;
  }).join("");

  // Report Body
  const body = $("#report-body");
  const fw = FRAMEWORK_META[state.selectedFramework];

  let actionsHtml = report.action_items.map((a, i) => {
    const prioColors = { high: "var(--accent-red)", medium: "var(--accent-amber)", low: "var(--accent-green)" };
    const prioLabels = { high: "높음", medium: "중간", low: "낮음" };
    return `<li><strong style="color:${prioColors[a.priority]}">[${prioLabels[a.priority]}]</strong> <strong>${a.title}</strong> — ${a.description}</li>`;
  }).join("");

  let risksHtml = report.risks.map(r =>
    `<li><strong>${r.risk}</strong> → ${r.mitigation}</li>`
  ).join("");

  body.innerHTML = `
    <h3>${fw.icon} 핵심 구조 분석</h3>
    <p>${report.structure_analysis}</p>

    <h3>📋 실행 전략 (Action Items)</h3>
    <ul>${actionsHtml}</ul>

    <div class="leverage-box">
      <h4>⚡ 레버리지 포인트</h4>
      <p><strong>${report.leverage_point.title}</strong></p>
      <p>${report.leverage_point.description}</p>
    </div>

    <div class="risk-box">
      <h4>⚠️ 리스크 분석</h4>
      <ul>${risksHtml}</ul>
    </div>

    <h3>📌 최종 결론</h3>
    <p><strong>${report.conclusion}</strong></p>
  `;
}

// ─── Utility Events ────────────────────────────────────────
$("#restart-btn").addEventListener("click", () => {
  state = { problem: "", selectedFramework: null, frameworkScores: {}, questions: [], answers: [], selectionReason: "" };
  $("#problem-input").value = "";
  showScreen("problem");
});

$("#copy-btn").addEventListener("click", () => {
  const reportText = $("#report-body").innerText;
  const scoresText = $("#score-dashboard").innerText;
  navigator.clipboard.writeText(`[Insight Engine 전략 리포트]\n\n${scoresText}\n\n${reportText}`)
    .then(() => alert("리포트가 클립보드에 복사되었습니다."));
});
