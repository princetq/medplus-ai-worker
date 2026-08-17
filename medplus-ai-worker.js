/**
 * MedPlus AI Pro - Cloudflare Worker
 * Version: 1.5.5
 *
 * Secrets / vars expected in Cloudflare:
 *   GEMINI_API_KEYS   = key1,key2,key3               (SECRET)
 *   NCBI_API_KEY      = your NCBI key                (SECRET, optional)
 *   NCBI_EMAIL        = maintainer@example.com       (VAR)
 *   NCBI_TOOL         = MedPlusAIPro                  (VAR)
 *   ALLOWED_ORIGINS   = https://ngocanh.io.vn,https://www.ngocanh.io.vn (VAR)
 *   GEMINI_MODELS     = gemini-3.6-flash,gemini-3.5-flash,gemini-3.5-flash-lite (VAR, optional)
 *
 * Design principle: this is NOT an open Gemini proxy. The client can only invoke
 * hard-coded clinical tasks below, which reduces key abuse risk.
 */

const DEFAULT_MODELS = [
  'gemini-3.6-flash',
  'gemini-3.5-flash',
  'gemini-3.5-flash-lite'
];

const MAX_REQUEST_BYTES = 900_000;
const MAX_EVIDENCE_ITEMS = 28;
const MAX_EVIDENCE_CHARS = 46_000;
const MAX_HISTORY_CHARS = 2_800;
const VISUAL_READER_REVISION = 'v071-full-table-bbox-1';

function json(data, status = 200, headers = {}) {
  return new Response(JSON.stringify(data), {
    status,
    headers: { 'content-type': 'application/json; charset=utf-8', ...headers }
  });
}

function csv(value) {
  return String(value || '').split(',').map(s => s.trim()).filter(Boolean);
}

function allowedOrigin(request, env) {
  const origin = request.headers.get('Origin') || '';
  const allowed = csv(env.ALLOWED_ORIGINS);
  if (!origin) return allowed[0] || '*';
  if (!allowed.length) return origin;
  if (allowed.includes(origin)) return origin;
  if (/^http:\/\/(localhost|127\.0\.0\.1)(:\d+)?$/.test(origin)) return origin;
  return '';
}

function cors(origin) {
  return {
    'Access-Control-Allow-Origin': origin || 'null',
    'Vary': 'Origin',
    'Access-Control-Allow-Methods': 'GET,POST,OPTIONS',
    'Access-Control-Allow-Headers': 'Content-Type',
    'Access-Control-Max-Age': '86400',
    'Cache-Control': 'no-store'
  };
}

async function readJsonBody(request, maxBytes = MAX_REQUEST_BYTES) {
  const len = Number(request.headers.get('content-length') || 0);
  if (len && len > maxBytes) throw new Error('REQUEST_TOO_LARGE');
  const text = await request.text();
  if (text.length > maxBytes) throw new Error('REQUEST_TOO_LARGE');
  try { return JSON.parse(text || '{}'); }
  catch { throw new Error('INVALID_JSON'); }
}

function clip(value, max) {
  const s = String(value == null ? '' : value);
  return s.length > max ? s.slice(0, max) + '\n[…đã rút gọn…]' : s;
}

function normalizeEvidence(raw) {
  if (!Array.isArray(raw)) return [];
  let used = 0;
  const out = [];
  for (const item of raw.slice(0, MAX_EVIDENCE_ITEMS)) {
    if (!item || !item.source_id || !item.text) continue;
    const remaining = MAX_EVIDENCE_CHARS - used;
    if (remaining <= 0) break;
    const text = clip(item.text, Math.min(3200, remaining));
    used += text.length;
    out.push({
      source_id: String(item.source_id),
      source_type: String(item.source_type || ''),
      label: clip(item.label || item.source_id, 300),
      text,
      url: clip(item.url || '', 800),
      metadata: item.metadata && typeof item.metadata === 'object' ? item.metadata : {}
    });
  }
  return out;
}

function getKeys(env) {
  return csv(env.GEMINI_API_KEYS || env.GEMINI_API_KEY);
}

function getModels(env) {
  const configured = csv(env.GEMINI_MODELS);
  return configured.length ? configured : DEFAULT_MODELS;
}

function getBuilderModels(env) {
  const configured = csv(env.GEMINI_BUILDER_MODELS);
  return configured.length ? configured : ['gemini-3.5-flash-lite'];
}

function extractGeminiText(data) {
  const parts = data?.candidates?.[0]?.content?.parts || [];
  return parts.map(p => p?.text || '').join('').trim();
}

async function callGemini(env, { system, prompt, userParts = null, temperature = 0.12, maxOutputTokens = 8192, jsonMode = true, modelsOverride = null, maxAttemptsOverride = null, mediaResolution = null }) {
  const keys = getKeys(env);
  if (!keys.length) throw new Error('GEMINI_KEY_NOT_CONFIGURED');
  const models = Array.isArray(modelsOverride) && modelsOverride.length ? modelsOverride : getModels(env);
  const errors = [];
  const configuredAttemptBudget=Number(maxAttemptsOverride ?? env.GEMINI_MAX_ATTEMPTS_PER_TASK ?? 0);
  const maxAttempts=configuredAttemptBudget>0?Math.max(1,Math.min(24,configuredAttemptBudget)):Math.max(1,Math.min(24,keys.length*models.length));
  let attempts = 0;

  // Free-tier: vẫn fallback khi lỗi tạm thời nhưng giới hạn số model request thực tế cho mỗi task.
  fallbackLoop: for (let keyIndex = 0; keyIndex < keys.length; keyIndex++) {
    const key = keys[keyIndex];
    for (let modelIndex = 0; modelIndex < models.length; modelIndex++) {
      const model = models[modelIndex];
      try {
        if (attempts >= maxAttempts) break fallbackLoop;
        attempts++;
        const url = `https://generativelanguage.googleapis.com/v1beta/models/${encodeURIComponent(model)}:generateContent`;
        const payload = {
          system_instruction: { parts: [{ text: system }] },
          contents: [{ role: 'user', parts: Array.isArray(userParts) && userParts.length ? userParts : [{ text: prompt }] }],
          generationConfig: {
            temperature,
            maxOutputTokens,
            ...(mediaResolution ? { mediaResolution } : {}),
            ...(jsonMode ? { responseMimeType: 'application/json' } : {})
          }
        };
        const res = await fetch(url, {
          method: 'POST',
          headers: {
            'content-type': 'application/json',
            'x-goog-api-key': key
          },
          body: JSON.stringify(payload)
        });
        const text = await res.text();
        let data = {};
        try { data = JSON.parse(text); } catch { data = { raw: text }; }
        if (!res.ok) {
          const message = data?.error?.message || text.slice(0, 180);
          errors.push({ key: keyIndex + 1, model, status: res.status, message });
          if (res.status === 429 && /prepayment credits are depleted|credit balance.*(?:0|depleted)|buy credits/i.test(message)) {
            const error = new Error('GEMINI_BILLING_PREPAY_DEPLETED');
            error.details = errors.slice(-12);error.attempts=attempts;
            error.terminalGeminiRequest = true;
            throw error;
          }
          // 400/401/403 are usually request/auth/location errors and retrying every model/key only wastes quota.
          // 404 can be model-specific, while 408/429/5xx are transient/fallback-worthy.
          if ([400,401,403].includes(res.status)) {
            const error = new Error('GEMINI_REQUEST_REJECTED');
            error.details = errors.slice(-12);error.attempts=attempts;
            error.terminalGeminiRequest = true;
            throw error;
          }
          continue;
        }
        const output = extractGeminiText(data);
        if (!output) {
          errors.push({ key: keyIndex + 1, model, status: 200, message: 'EMPTY_MODEL_OUTPUT' });
          continue;
        }
        let parsed = output;
        if (jsonMode) {
          try { parsed = JSON.parse(output); }
          catch {
            const cleaned = output.replace(/^```json\s*/i, '').replace(/```\s*$/i, '').trim();
            parsed = JSON.parse(cleaned);
          }
        }
        return { data: parsed, model, key_index: keyIndex + 1, usage: data?.usageMetadata || {}, attempts };
      } catch (err) {
        if (err?.terminalGeminiRequest) throw err;
        errors.push({ key: keyIndex + 1, model, status: 0, message: String(err?.message || err) });
      }
    }
  }
  const error = new Error('ALL_GEMINI_FALLBACKS_FAILED');
  error.details = errors.slice(-12);error.attempts=attempts;
  throw error;
}

const CASE_PARSE_SYSTEM = `Bạn là Clinical Case Parser của MedPlus AI Pro.
NHIỆM VỤ DUY NHẤT: chuyển câu hỏi tự nhiên và dữ kiện bệnh nhân thành dữ liệu có cấu trúc. KHÔNG tư vấn, KHÔNG tính liều, KHÔNG suy diễn dữ kiện không được cung cấp.

QUY TẮC:
- Nếu có PRIOR_CASE_JSON, coi đó là hồ sơ ca bệnh đang theo dõi. Hãy TRẢ VỀ TOÀN BỘ HỒ SƠ ĐÃ HỢP NHẤT sau khi áp dụng dữ kiện mới.
- Dữ kiện mới được phép bổ sung hoặc ghi đè dữ kiện cũ CHỈ khi người dùng/file mới nói rõ giá trị mới. Nếu không nhắc lại, phải giữ nguyên dữ kiện cũ.
- Nếu người dùng nói một thuốc đã ngừng/đổi liều/đổi đường dùng, cập nhật đúng thuốc đó; không xóa các thuốc khác.
- Với xét nghiệm lặp lại theo thời gian, giữ giá trị mới làm current value và thêm/giữ trend khi có thời điểm.
- Không tự xóa chẩn đoán, dị ứng, thuốc hay dữ kiện trước đó chỉ vì lượt mới không nhắc đến chúng.
- Chỉ lấy dữ kiện xuất hiện trong câu hỏi, PATIENT_CONTEXT hoặc PRIOR_CASE_JSON. Không tự điền giá trị bình thường/ước đoán.
- Nếu chỉ có năm sinh (ví dụ "sinh năm 2014"), đặt birth_year=2014 và age_years=null; không tự chọn ngày/tháng sinh.
- Phân biệt thuốc đang dùng (current), bác sĩ dự định thêm (planned), đã ngừng (stopped) nếu câu nói thể hiện rõ.
- Giữ nguyên liều, tần suất và đường dùng quan trọng. Có thể chuẩn hóa đơn vị hiển nhiên (µmol/L -> umol/L), nhưng KHÔNG tạo giá trị dẫn xuất.
- Nếu người dùng/hồ sơ cung cấp trực tiếp ClCr/creatinine clearance, ghi vào renal.reported_crcl; nếu cung cấp eGFR, ghi vào renal.reported_egfr. Không tự tính một chỉ số này từ chỉ số kia trong Parser.
- Nếu cùng một dữ kiện có nhiều giá trị theo thời gian, ghi vào trend/labs với thời điểm nếu có.
- Nếu một dữ kiện từ file có confidence thấp/moderate, giữ mức không chắc chắn trong uncertainty; không nâng confidence.
- Chẩn đoán, dấu hiệu, xét nghiệm và thuốc đều là dữ kiện ca bệnh, không phải bằng chứng khuyến cáo.
- Nếu không biết giới, chiều cao, creatinin... đặt null/[]; không bịa.

JSON BẮT BUỘC:
{
  "is_clinical_case": true,
  "patient": {
    "sex": "male|female|unknown",
    "age_years": null,
    "birth_year": null,
    "birth_date": null,
    "weight_kg": null,
    "height_cm": null,
    "dry_weight_kg": null,
    "icu": null,
    "edema": null,
    "ascites": null,
    "sarcopenia": null,
    "amputation": {"present": null, "details": ""}
  },
  "renal": {
    "serum_creatinine": {"value": null, "unit": null, "time": ""},
    "cystatin_c_mg_l": null,
    "reported_egfr": {"value": null, "unit": ""},
    "reported_crcl": {"value": null, "unit": "", "method": "", "time": ""},
    "creatinine_trend": [{"value": 0, "unit": "mg/dL|umol/L", "hours_ago": null, "days_ago": null, "time": ""}],
    "urine_output_ml_kg_h": null,
    "urine_output_duration_h": null,
    "dialysis": "none|IHD|CRRT|PD|unknown",
    "crrt": {"mode":"CVVH|CVVHD|CVVHDF|unknown","effluent_ml_kg_h":null,"dialysate_ml_h":null,"replacement_ml_h":null}
  },
  "hepatic": {
    "cirrhosis": null,
    "bilirubin_mg_dl": null,
    "albumin_g_dl": null,
    "inr": null,
    "ascites_grade": "none|mild|moderate_severe|unknown",
    "encephalopathy_grade": "none|grade1_2|grade3_4|unknown"
  },
  "diagnoses": [],
  "medications": [{"name":"","dose":"","frequency":"","route":"","status":"current|planned|stopped|unknown"}],
  "allergies": [],
  "labs": [{"name":"","value":"","unit":"","time":"","confidence":"high|moderate|low"}],
  "tasks": [{"key":"dose|interaction|compatibility|renal|hepatic|monitoring|indication|toxicity|other","question":""}],
  "uncertainty": [],
  "missing_obvious": []
}`;

async function handleParseCase(body, env) {
  const question = clip(body.question || '', 9000);
  if (!question) throw new Error('QUESTION_REQUIRED');
  const patientContext = clip(body.patient_context || '', 16000);
  const priorCase = body.prior_case && typeof body.prior_case === 'object' ? body.prior_case : null;
  const priorCaseText = priorCase ? clip(JSON.stringify(priorCase), 22000) : '';
  const resolvedDrugs = Array.isArray(body.resolved_drugs) ? body.resolved_drugs.slice(0, 16) : [];
  const currentDate = clip(body.current_date || new Date().toISOString().slice(0, 10), 40);
  const prompt = `NGÀY HIỆN TẠI: ${currentDate}\n\nPRIOR_CASE_JSON (hồ sơ ca bệnh hiện tại, nếu có):\n${priorCaseText || '(không có - đây là ca mới)'}\n\nDỮ KIỆN/CÂU HỎI MỚI CẦN HỢP NHẤT:\n${question}\n\nPATIENT_CONTEXT TỪ FILE NẾU CÓ:\n${patientContext || '(không có)'}\n\nTHUỐC DRUG RESOLVER ĐÃ NHẬN DIỆN (chỉ để chuẩn hóa tên, không phải dữ kiện bổ sung):\n${JSON.stringify(resolvedDrugs)}`;
  return callGemini(env, { system: CASE_PARSE_SYSTEM, prompt, temperature: 0.01, maxOutputTokens: 4500, jsonMode: true });
}




const QUESTION_COMPILER_SYSTEM = `Bạn là AI Question Compiler của MedPlus AI Pro.
Bạn KHÔNG trả lời câu hỏi y khoa. Bạn biên dịch câu hỏi tự nhiên thành một kế hoạch tìm kiếm tổng quát áp dụng cho MỌI loại câu hỏi, không dựa vào danh sách câu mẫu.

MỤC TIÊU:
- Xác định người dùng hỏi về một thuốc cụ thể, nhiều thuốc, một nhóm/đặc tính, hay toàn bộ danh mục bệnh viện.
- Mở rộng tên biệt dược/hoạt chất/nhóm/thuật ngữ lâm sàng thành các search_terms đủ recall nhưng không bịa danh sách thuốc.
- Xác định các mục nguồn cần kiểm tra: liều, suy thận, ADR, tương tác, tương kỵ, thai kỳ, chỉ định, dược động học, pha truyền...
- Xác định visual_topics có khả năng nằm trong bảng/hình.
- Xác định patient_metrics cần dùng (ClCr, eGFR, cân nặng, tuổi, BSA, Child-Pugh, CRRT...) nhưng KHÔNG tự tính.
- Quyết định có cần Clinical Case Parser hay không. Câu hỏi kiến thức/danh mục không có dữ kiện người bệnh thì requires_clinical_parser=false. Nếu có hồ sơ ca trước nhưng câu mới rõ ràng là câu hỏi chung/toàn kho và không nói “bệnh nhân này/ca này”, case_context_role=ignore để không làm nhiễu; KHÔNG xóa ca. Nếu câu mới bổ sung dữ kiện cho ca thì case_context_role=merge. Nếu chỉ dùng lại ca mà không có dữ kiện mới thì case_context_role=reference.
- Với câu hỏi dạng “thuốc nào / những thuốc / có những gì / toàn bộ / bệnh viện có...”, scope phải là exhaustive hoặc broad_inventory; không được giả định vài candidate đầu là đủ.
- Với câu hỏi về một đặc tính có thể xuất hiện ở nhiều thuốc (ADR, QT, chỉnh liều thận, tương tác, pha truyền...), nếu người dùng hỏi toàn danh mục thì phải yêu cầu GLOBAL search cả Dược thư + HDSD + visual index.
- Không đưa khuyến cáo y khoa.

JSON:
{
  "scope":"targeted|multi_drug|broad_inventory|exhaustive",
  "answer_contract":"direct|complete_list|comparison|patient_specific_dose|evidence_review|other",
  "entity_terms":["tên/hoạt chất/nhóm người dùng nêu"],
  "search_terms":["term mở rộng để tìm trong toàn kho"],
  "negative_terms":[],
  "section_hints":["tên mục nguồn cần ưu tiên"],
  "visual_topics":["dose|renal_dose|adr|interaction|compatibility|monitoring|other"],
  "patient_metrics_needed":["ClCr|eGFR|weight|age|BSA|Child-Pugh|CRRT|..."],
  "requires_clinical_parser":false,
  "case_context_role":"merge|reference|ignore",
  "must_search_global_graph":true,
  "must_search_hdsd_corpus":true,
  "must_check_visual_index":true,
  "must_be_exhaustive":false,
  "reason":"..."
}`;

async function handleQuestionCompiler(body, env) {
  const question = clip(body.question || '', 8000);
  if (!question) throw new Error('QUESTION_REQUIRED');
  const clinical = clip(body.clinical_context || body.patient_context_preview || '', 12000);
  const priorCase = clip(body.prior_case_summary || '', 9000);
  const hasPriorCase = !!body.has_prior_case;
  const resolved = Array.isArray(body.resolved_drugs) ? body.resolved_drugs.slice(0, 20) : [];
  const prompt = `CÂU HỎI:\n${question}\n\nPATIENT/FILE CONTEXT PREVIEW (chưa parse, có thể trống):\n${clinical || '(không có)'}\n\nHAS_PRIOR_CASE: ${hasPriorCase}\nPRIOR_CASE_SUMMARY (chỉ để quyết định có áp dụng ca vào câu mới hay không):\n${priorCase || '(không có)'}\n\nDRUG RESOLVER CANDIDATES (chỉ là gợi ý, có thể chưa đầy đủ):\n${JSON.stringify(resolved)}\n\nHãy biên dịch thành kế hoạch tìm kiếm tổng quát. Nếu câu hỏi yêu cầu danh sách/tập hợp, phải ưu tiên recall và completeness. Search terms nên phản ánh thuật ngữ thực tế có thể xuất hiện trong HDSD/Dược thư, gồm biến thể chính tả/đồng nghĩa cần thiết, nhưng không tự bịa tên thuốc vào kết quả.`;
  return callGemini(env, { system: QUESTION_COMPILER_SYSTEM, prompt, temperature: 0.01, maxOutputTokens: 2200, jsonMode: true });
}

const COMPLETENESS_SYSTEM = `Bạn là AI Completeness Checker của MedPlus AI Pro.
Bạn KHÔNG trả lời câu hỏi y khoa. Bạn kiểm tra xem retrieval hiện tại có nguy cơ bỏ sót nguồn/thuốc/mục/bảng nào cần thiết để trả lời đúng câu hỏi hay không.

Bạn nhận QUESTION PLAN + RETRIEVAL STATS + danh sách candidate/evidence metadata.
QUY TẮC:
- Không coi top-k retrieval là “đủ” chỉ vì có vài bằng chứng tốt.
- Nếu answer_contract=complete_list hoặc scope=exhaustive, chỉ complete=true khi corpus được quét theo phạm vi toàn bộ cần thiết và visual coverage đủ cho predicate có thể nằm trong ảnh/bảng.
- Kiểm tra độc lập các nguồn MedPlus, HDSD, Dược thư, visual index; nếu một nguồn được ưu tiên nhưng chưa quét thì phải yêu cầu expansion.
- Nếu một khái niệm có synonym/thuật ngữ khác có thể làm bỏ sót, thêm search_terms cụ thể nhưng không tự bịa tên thuốc vào kết quả.
- Với patient-specific dose, yêu cầu đúng metric mà evidence dùng; nếu đã có ClCr/eGFR phù hợp thì không được để pipeline rơi về liều chung.
- Nếu visual index chưa complete và câu hỏi exhaustive về liều/ADR/tương tác/tương kỵ/monitoring, complete=false.
- Tối đa đề xuất 12 search_terms mới và 8 section_hints mới.

JSON:
{
  "complete":true,
  "coverage_score":0,
  "missing_search_terms":[],
  "missing_section_hints":[],
  "must_expand_global_graph":false,
  "must_expand_hdsd":false,
  "must_expand_visuals":false,
  "must_include_source_types":[],
  "gaps":[],
  "reason":"..."
}`;

async function handleCompletenessCheck(body, env) {
  const question = clip(body.question || '', 8000);
  const plan = body.plan && typeof body.plan === 'object' ? body.plan : {};
  const stats = body.stats && typeof body.stats === 'object' ? body.stats : {};
  const candidates = Array.isArray(body.candidates) ? body.candidates.slice(0, 80).map(x => ({
    source_id: clip(x.source_id || '', 220), source_type: clip(x.source_type || '', 80), label: clip(x.label || '', 360),
    drug_keys: x.metadata?.drug_keys || [], section: clip(x.metadata?.section || '', 220)
  })) : [];
  const prompt = `QUESTION:\n${question}\n\nQUESTION_PLAN:\n${JSON.stringify(plan)}\n\nRETRIEVAL_STATS:\n${JSON.stringify(stats)}\n\nCANDIDATE/EVIDENCE METADATA:\n${JSON.stringify(candidates)}`;
  return callGemini(env, { system: COMPLETENESS_SYSTEM, prompt, temperature: 0.0, maxOutputTokens: 2400, jsonMode: true });
}

const RETRIEVAL_DIRECTOR_SYSTEM = `Bạn là AI Retrieval Director của MedPlus AI Pro.
Bạn KHÔNG trả lời câu hỏi y khoa. Bạn điều phối việc lấy bằng chứng.
Mục tiêu cao nhất: khai thác HẾT dữ liệu nội bộ liên quan trước khi dùng nguồn ngoài.

QUY TẮC:
- MedPlus drug master giúp nhận diện đúng sản phẩm/hoạt chất.
- HDSD bệnh viện và Dược thư Quốc gia là nguồn ưu tiên cao nhất cho dữ liệu thuốc hiện có.
- Nếu nguồn có bảng/hình, phải yêu cầu đọc visual; không được giả định text JSON đã chứa nội dung bảng.
- Với liều, ADR, hiệu chỉnh thận/gan, tương tác, tương kỵ, pha truyền, theo dõi: luôn kiểm tra khả năng dữ liệu nằm trong bảng/hình.
- PubMed/PMC chỉ là lớp bổ sung khi nguồn nội bộ không đủ, cần bằng chứng mới, hoặc cần xử lý xung đột.
- Không được loại một thuốc chỉ vì tên biệt dược khác; phải dựa hoạt chất/canonical drug key.
- Không đưa khuyến cáo, chỉ trả kế hoạch retrieval.

JSON:
{
  "local_priority":["HDSD_BV","DUOC_THU","MEDPLUS"],
  "focus_topics":["dose|renal|adr|interaction|compatibility|monitoring|indication|other"],
  "section_hints":["..."],
  "must_read_visuals":true,
  "must_scan_all_visuals_for_resolved_drugs":true,
  "pubmed_role":"supplement_only|needed_for_gap|not_needed",
  "reason":"..."
}`;

async function handleRetrievalDirector(body, env) {
  const question = clip(body.question || '', 7000);
  if (!question) throw new Error('QUESTION_REQUIRED');
  const clinical = clip(body.clinical_context || '', 12000);
  const drugs = Array.isArray(body.drugs) ? body.drugs.slice(0, 16) : [];
  const intents = Array.isArray(body.intents) ? body.intents.slice(0, 20) : [];
  const prompt = `CÂU HỎI:\n${question}\n\nCLINICAL CONTEXT:\n${clinical || '(không có)'}\n\nTHUỐC/HOẠT CHẤT ĐÃ RESOLVE:\n${JSON.stringify(drugs)}\n\nINTENTS TỪ ROUTER:\n${JSON.stringify(intents)}\n\nHãy lập kế hoạch retrieval; mặc định phải đọc visual của toàn bộ nguồn đúng thuốc nếu có.`;
  return callGemini(env, { system: RETRIEVAL_DIRECTOR_SYSTEM, prompt, temperature: 0.02, maxOutputTokens: 1400, jsonMode: true });
}

const VISUAL_EVIDENCE_SYSTEM = `Bạn là Visual Evidence Reader của MedPlus AI Pro.
NHIỆM VỤ DUY NHẤT: đọc CHÍNH XÁC từng ảnh/bảng từ HDSD hoặc Dược thư và biến thành các evidence unit có thể trích dẫn. KHÔNG tư vấn điều trị và KHÔNG dùng kiến thức ngoài ảnh.

QUY TẮC BẮT BUỘC:
- Đọc TOÀN BỘ bảng/hình; không chỉ đọc ô có vẻ liên quan câu hỏi.
- Giữ chính xác tiêu đề cột/hàng, liều, đơn vị, ngưỡng ClCr/eGFR, khoảng tuổi/cân nặng, tần suất ADR, dấu *, chú thích/footnote.
- Không tự sửa số liệu hoặc chuẩn hóa làm mất ý nghĩa. Nếu không đọc rõ, đánh confidence thấp và ghi unreadable_text.
- Với bảng liều: mỗi hàng/nhánh liều lâm sàng phải là một evidence unit riêng nếu có thể.
- Với ADR: chia theo nhóm/tần suất hoặc hàng lâm sàng hợp lý để citation có thể focus đúng vùng.
- Với ảnh không phải bảng nhưng có nội dung y khoa, chia thành các vùng có ý nghĩa.
- bbox_norm = [x,y,width,height] trên thang 0..1000, bao đúng vùng ảnh chứa evidence unit. Nếu không xác định được, dùng [0,0,1000,1000].
- Mỗi evidence unit phải tự đủ nghĩa và chứa header cần thiết để hiểu hàng đó.
- source_id phải trả đúng source_id đầu vào cho ảnh; unit_id do bạn tạo ngắn gọn, duy nhất trong ảnh.

JSON:
{
  "items":[{
    "source_id":"ID_ẢNH_ĐẦU_VÀO",
    "visual_type":"dose_table|adr_table|interaction_table|compatibility_table|monitoring_table|figure|other",
    "title":"tiêu đề nhìn thấy hoặc mô tả trung tính",
    "complete_read":true,
    "unreadable_text":["..."],
    "units":[{
      "unit_id":"u01",
      "topic":"dose|renal_dose|hepatic_dose|pediatric_dose|adr|interaction|compatibility|monitoring|other",
      "text":"nội dung đầy đủ, giữ số liệu/đơn vị/header cần thiết",
      "bbox_norm":[0,0,1000,1000],
      "confidence":"high|moderate|low"
    }]
  }]
}`;


function bytesToBase64Worker(bytes) {
  let out = '';
  const step = 0x8000;
  for (let i = 0; i < bytes.length; i += step) out += String.fromCharCode(...bytes.subarray(i, Math.min(bytes.length, i + step)));
  return btoa(out);
}
function isPublicVisualUrl(raw) {
  try {
    const u = new URL(String(raw || ''));
    if (u.protocol !== 'https:') return false;
    const h = u.hostname.toLowerCase();
    if (!h || h === 'localhost' || h.endsWith('.local')) return false;
    if (/^(127\.|10\.|0\.|169\.254\.|192\.168\.|172\.(1[6-9]|2\d|3[01])\.)/.test(h)) return false;
    if (h === '::1' || h.startsWith('fc') || h.startsWith('fd')) return false;
    return true;
  } catch { return false; }
}
async function fetchVisualUrlAsBase64(rawUrl, maxBytes = 5 * 1024 * 1024) {
  if (!isPublicVisualUrl(rawUrl)) throw new Error('VISUAL_URL_NOT_ALLOWED');
  const res = await fetch(rawUrl, { redirect: 'follow', headers: { 'user-agent': 'MedPlusAIPro-Visual/1.0.0' } });
  if (!res.ok) throw new Error(`VISUAL_FETCH_HTTP_${res.status}`);
  const len = Number(res.headers.get('content-length') || 0);
  if (len && len > maxBytes) throw new Error('VISUAL_URL_TOO_LARGE');
  const mime = String(res.headers.get('content-type') || '').split(';')[0].toLowerCase();
  if (!['image/png','image/jpeg','image/webp'].includes(mime)) throw new Error('VISUAL_URL_BAD_MIME');
  const buf = await res.arrayBuffer();
  if (buf.byteLength > maxBytes) throw new Error('VISUAL_URL_TOO_LARGE');
  return { mime, data: bytesToBase64Worker(new Uint8Array(buf)), sha256: await sha256HexWorker(buf), content_length: buf.byteLength, etag:res.headers.get('etag')||'', last_modified:res.headers.get('last-modified')||'' };
}

function bytesToHex(bytes) {
  return Array.from(bytes, b => b.toString(16).padStart(2, '0')).join('');
}
async function sha256HexWorker(buf) {
  const digest = await crypto.subtle.digest('SHA-256', buf);
  return bytesToHex(new Uint8Array(digest));
}
function normalizeHttpValidator(v) { return String(v || '').trim(); }
function validatorCanReuse(previous, meta) {
  if (!previous?.sha256) return '';
  const pe = normalizeHttpValidator(previous.etag), e = normalizeHttpValidator(meta.etag);
  // Strong or weak ETag equality is still useful here as a cache validator supplied by the origin.
  if (pe && e && pe === e) return 'etag';
  const pl = normalizeHttpValidator(previous.last_modified), l = normalizeHttpValidator(meta.last_modified);
  const pn = Number(previous.content_length || 0), n = Number(meta.content_length || 0);
  if (pl && l && pl === l && pn > 0 && n > 0 && pn === n) return 'last-modified+length';
  return '';
}
async function fingerprintOneVisual(im, maxBytes = 8 * 1024 * 1024) {
  const rawUrl = String(im?.image_url || '');
  if (!isPublicVisualUrl(rawUrl)) throw new Error('VISUAL_URL_NOT_ALLOWED');
  const headers = { 'user-agent': 'MedPlusAIPro-Fingerprint/1.0.0' };
  let meta = { etag:'', last_modified:'', content_length:0, mime:'' };
  try {
    const head = await fetch(rawUrl, { method:'HEAD', redirect:'follow', headers });
    if (head.ok) {
      meta.etag = head.headers.get('etag') || '';
      meta.last_modified = head.headers.get('last-modified') || '';
      meta.content_length = Number(head.headers.get('content-length') || 0);
      meta.mime = String(head.headers.get('content-type') || '').split(';')[0].toLowerCase();
      const via = validatorCanReuse(im.previous || {}, meta);
      if (via) return { source_id:String(im.source_id||''), ok:true, sha256:String(im.previous.sha256), ...meta, unchanged:true, unchanged_via:via, content_fetched:false };
    }
  } catch {}
  const res = await fetch(rawUrl, { redirect:'follow', headers });
  if (!res.ok) throw new Error(`VISUAL_FETCH_HTTP_${res.status}`);
  const mime = String(res.headers.get('content-type') || meta.mime || '').split(';')[0].toLowerCase();
  if (!['image/png','image/jpeg','image/webp'].includes(mime)) throw new Error('VISUAL_URL_BAD_MIME');
  const len = Number(res.headers.get('content-length') || 0);
  if (len && len > maxBytes) throw new Error('VISUAL_URL_TOO_LARGE');
  const buf = await res.arrayBuffer();
  if (buf.byteLength > maxBytes) throw new Error('VISUAL_URL_TOO_LARGE');
  return {
    source_id:String(im.source_id||''), ok:true, sha256:await sha256HexWorker(buf), mime,
    content_length:buf.byteLength,
    etag:res.headers.get('etag') || meta.etag || '',
    last_modified:res.headers.get('last-modified') || meta.last_modified || '',
    unchanged:false, unchanged_via:'sha256', content_fetched:true
  };
}
async function handleImageFingerprint(body) {
  const images = Array.isArray(body?.images) ? body.images.slice(0, 16) : [];
  const items = [];
  for (const im of images) {
    try { items.push(await fingerprintOneVisual(im)); }
    catch (e) { items.push({ source_id:String(im?.source_id||''), ok:false, error:String(e?.message||e) }); }
  }
  return { data:{ algorithm:'sha256', visual_reader_revision:VISUAL_READER_REVISION, items } };
}


function base64ToBytesWorker(data) {
  const bin = atob(String(data || '').replace(/\s/g, ''));
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return bytes;
}
async function sha256TextWorker(text) {
  return sha256HexWorker(new TextEncoder().encode(String(text || '')));
}
async function visualCacheSourceKey(sourceId) {
  return `vc:s:${VISUAL_READER_REVISION}:${await sha256TextWorker(sourceId)}`;
}
function visualCacheHashKey(sha256) {
  return `vc:h:${VISUAL_READER_REVISION}:${String(sha256 || '').toLowerCase()}`;
}
function visualCacheSafeRecord(record) {
  return {
    schema_version:'1.0',
    visual_reader_revision:VISUAL_READER_REVISION,
    source_id:String(record.source_id||''),
    source_type:String(record.source_type||''),
    label:clip(record.label||'',500),
    image_url:clip(record.image_url||'',1600),
    metadata:record.metadata&&typeof record.metadata==='object'?record.metadata:{},
    sha256:String(record.sha256||''),
    etag:String(record.etag||''),
    last_modified:String(record.last_modified||''),
    content_length:Number(record.content_length||0),
    visual_type:String(record.visual_type||''),
    title:clip(record.title||'',500),
    complete_read:record.complete_read!==false,
    unreadable_text:Array.isArray(record.unreadable_text)?record.unreadable_text.slice(0,30):[],
    units:Array.isArray(record.units)?record.units.slice(0,100).map(u=>({
      unit_id:clip(u.unit_id||'u',120),topic:clip(u.topic||'other',100),text:clip(u.text||'',6000),
      bbox_norm:Array.isArray(u.bbox_norm)&&u.bbox_norm.length===4?u.bbox_norm:[0,0,1000,1000],
      confidence:clip(u.confidence||'moderate',40)
    })):[],
    model:clip(record.model||'',120),
    indexed_at:record.indexed_at||new Date().toISOString()
  };
}
async function visualCachePut(env, asset, item, sha256, model='') {
  if (!env.VISUAL_CACHE || !sha256 || !item || item.complete_read===false || !Array.isArray(item.units) || !item.units.length) return false;
  if (!['HDSD_VISUAL','DUOC_THU_VISUAL'].includes(String(asset.source_type||''))) return false;
  const record = visualCacheSafeRecord({
    source_id:asset.source_id, source_type:asset.source_type, label:asset.label, image_url:asset.image_url,
    metadata:asset.metadata||{}, sha256, etag:asset.etag||'', last_modified:asset.last_modified||'', content_length:asset.content_length||0, visual_type:item.visual_type||'', title:item.title||'',
    complete_read:item.complete_read!==false, unreadable_text:item.unreadable_text||[], units:item.units||[], model
  });
  const value = JSON.stringify(record);
  await Promise.all([
    env.VISUAL_CACHE.put(await visualCacheSourceKey(asset.source_id), value),
    env.VISUAL_CACHE.put(visualCacheHashKey(sha256), value)
  ]);
  return true;
}
function visualCacheOverlay(record, asset) {
  if (!record) return null;
  return {
    ...record,
    source_id:String(asset.source_id||record.source_id||''),
    source_type:String(asset.source_type||record.source_type||''),
    label:String(asset.label||record.label||''),
    image_url:String(asset.image_url||record.image_url||''),
    metadata:{...(record.metadata||{}),...(asset.metadata||{})}
  };
}
async function visualCacheGetSource(env, asset) {
  if (!env.VISUAL_CACHE || !asset?.source_id) return null;
  const record = await env.VISUAL_CACHE.get(await visualCacheSourceKey(asset.source_id), 'json');
  if (!record || record.visual_reader_revision !== VISUAL_READER_REVISION || record.complete_read===false) return null;
  return visualCacheOverlay(record, asset);
}
async function visualCacheGetHash(env, asset, sha256) {
  if (!env.VISUAL_CACHE || !sha256) return null;
  const record = await env.VISUAL_CACHE.get(visualCacheHashKey(sha256), 'json');
  if (!record || record.visual_reader_revision !== VISUAL_READER_REVISION || record.complete_read===false) return null;
  const over = visualCacheOverlay(record, asset);
  // Tạo alias theo source hiện tại để các lần sau chỉ cần 1 KV read.
  try { await env.VISUAL_CACHE.put(await visualCacheSourceKey(asset.source_id), JSON.stringify(visualCacheSafeRecord(over))); } catch {}
  return over;
}
async function handleVisualCacheResolve(body, env) {
  const images = Array.isArray(body?.images) ? body.images.slice(0, 16) : [];
  if (!env.VISUAL_CACHE) return { data:{ enabled:false, hits:[], misses:images.map(x=>({source_id:String(x?.source_id||''),reason:'KV_NOT_BOUND'})) } };
  const hits=[],misses=[];
  for (const raw of images) {
    const asset={source_id:String(raw?.source_id||''),source_type:String(raw?.source_type||''),label:clip(raw?.label||'',500),image_url:String(raw?.image_url||''),metadata:raw?.metadata&&typeof raw.metadata==='object'?raw.metadata:{}};
    if (!asset.source_id) continue;
    try {
      const sourceKey=await visualCacheSourceKey(asset.source_id);
      let sourceRec=await env.VISUAL_CACHE.get(sourceKey,'json');
      if(sourceRec&&sourceRec.visual_reader_revision!==VISUAL_READER_REVISION)sourceRec=null;
      let sha=String(raw?.sha256||'').toLowerCase();

      // Builder đã có SHA-256: không fetch lại ảnh. Runtime: xác nhận source alias bằng HTTP validator/SHA trước khi dùng.
      if(sourceRec){
        if(/^[a-f0-9]{64}$/.test(sha)){
          if(sha===String(sourceRec.sha256||'').toLowerCase()){
            hits.push({source_id:asset.source_id,sha256:sha,item:visualCacheOverlay(sourceRec,asset),via:'source+sha256'});continue;
          }
        }else if(asset.image_url){
          const fp=await fingerprintOneVisual({...asset,previous:{sha256:sourceRec.sha256||'',etag:sourceRec.etag||'',last_modified:sourceRec.last_modified||'',content_length:sourceRec.content_length||0}});
          if(fp?.ok)sha=String(fp.sha256||'').toLowerCase();
          if(sha&&sha===String(sourceRec.sha256||'').toLowerCase()){
            hits.push({source_id:asset.source_id,sha256:sha,item:visualCacheOverlay(sourceRec,asset),via:`source+${fp.unchanged_via||'sha256'}`});continue;
          }
        }
      }

      if(!/^[a-f0-9]{64}$/.test(sha)&&asset.image_url){
        const fp=await fingerprintOneVisual(asset);if(fp?.ok)sha=String(fp.sha256||'').toLowerCase();
      }
      let rec=null;
      if(/^[a-f0-9]{64}$/.test(sha))rec=await visualCacheGetHash(env,asset,sha);
      if(rec)hits.push({source_id:asset.source_id,sha256:sha,item:rec,via:'sha256'});
      else misses.push({source_id:asset.source_id,sha256:sha||'',reason:sourceRec?'CONTENT_CHANGED_OR_HASH_MISS':'CACHE_MISS'});
    }catch(e){misses.push({source_id:asset.source_id,reason:String(e?.message||e)})}
  }
  return { data:{ enabled:true, visual_reader_revision:VISUAL_READER_REVISION, hits, misses } };
}

async function handleReadVisualEvidence(body, env) {
  const question = clip(body.question || '', 6000);
  const clinical = clip(body.clinical_context || '', 9000);
  const builderMode = body.builder_mode === true;
  const maxImages = builderMode ? 10 : 6;
  const images = Array.isArray(body.images) ? body.images.slice(0, maxImages) : [];
  if (!images.length) return { data: { items: [] }, model: '', key_index: 0, usage: {}, attempts: 0, visual_cache:{enabled:!!env.VISUAL_CACHE,stored:0} };

  const userParts = [{ text: `CÂU HỎI HIỆN TẠI (chỉ để biết ngữ cảnh; vẫn phải đọc toàn bộ ảnh):\n${question}\n\nCLINICAL CONTEXT:\n${clinical || '(không có)'}\n\nSau đây là ${images.length} ảnh. Hãy đọc đầy đủ từng ảnh.` }];
  const prepared = [];
  let totalBytes = 0;
  for (let i = 0; i < images.length; i++) {
    const im = images[i] || {};
    let mime = String(im.mime_type || '').toLowerCase();
    let data = String(im.data_base64 || '').replace(/\s/g, '');
    let sha256 = String(im.sha256||'').toLowerCase(),etag=String(im.etag||''),last_modified=String(im.last_modified||''),content_length=Number(im.content_length||0);
    if (!data && im.image_url) {
      const fetched = await fetchVisualUrlAsBase64(im.image_url, 5 * 1024 * 1024);
      mime = fetched.mime; data = fetched.data; sha256 = fetched.sha256 || '';etag=fetched.etag||'';last_modified=fetched.last_modified||'';content_length=fetched.content_length||0;
    } else if (data && !/^[a-f0-9]{64}$/.test(sha256)) {
      try { sha256 = await sha256HexWorker(base64ToBytesWorker(data)); } catch {}
    }
    if (!mime) mime = 'image/png';
    if (!['image/png','image/jpeg','image/webp'].includes(mime)) throw new Error('UNSUPPORTED_VISUAL_MIME');
    if (!data) throw new Error('VISUAL_DATA_REQUIRED');
    totalBytes += estimateBase64Bytes(data);
    if (totalBytes > 14 * 1024 * 1024) throw new Error('VISUAL_BATCH_TOO_LARGE');
    prepared.push({asset:{source_id:String(im.source_id||''),source_type:String(im.source_type||''),label:String(im.label||''),image_url:String(im.image_url||''),metadata:im.metadata||{},etag,last_modified,content_length},sha256});
    userParts.push({ text: `ẢNH ${i + 1}\nsource_id: ${clip(im.source_id || '', 260)}\nsource_type: ${clip(im.source_type || '', 80)}\nlabel: ${clip(im.label || '', 500)}\nmetadata: ${clip(JSON.stringify(im.metadata || {}), 3500)}` });
    userParts.push({ inline_data: { mime_type: mime, data } });
  }

  let out;
  if (builderMode) {
    const tier = String(body.model_tier || 'lite').toLowerCase();
    const models = tier === 'flash' ? ['gemini-3.5-flash'] : getBuilderModels(env);
    const requested = String(body.media_resolution || 'medium').toLowerCase();
    const mediaResolution = requested === 'high' ? 'MEDIA_RESOLUTION_HIGH' : requested === 'low' ? 'MEDIA_RESOLUTION_LOW' : 'MEDIA_RESOLUTION_MEDIUM';
    out = await callGemini(env, { system: VISUAL_EVIDENCE_SYSTEM, prompt: '', userParts, temperature: 0.0, maxOutputTokens: 9000, jsonMode: true, modelsOverride: models, maxAttemptsOverride: 1, mediaResolution });
  } else {
    out = await callGemini(env, { system: VISUAL_EVIDENCE_SYSTEM, prompt: '', userParts, temperature: 0.0, maxOutputTokens: 9000, jsonMode: true, modelsOverride:getBuilderModels(env), maxAttemptsOverride:1, mediaResolution:'MEDIA_RESOLUTION_MEDIUM' });
  }

  let stored=0;
  if (body.cache_visual_results !== false && env.VISUAL_CACHE) {
    for (const item of (out.data?.items || [])) {
      const p=prepared.find(x=>x.asset.source_id===item.source_id);
      if (!p?.sha256 || item.complete_read===false) continue;
      try { if(await visualCachePut(env,p.asset,item,p.sha256,out.model||'')) stored++; } catch(e) { console.warn('VISUAL_CACHE_PUT_FAILED',e); }
    }
  }
  return {...out, visual_cache:{enabled:!!env.VISUAL_CACHE,stored}};
}

const EVIDENCE_SELECTOR_SYSTEM = `Bạn là AI Evidence Selector của MedPlus AI Pro.
Bạn KHÔNG trả lời câu hỏi. Bạn chọn bằng chứng phù hợp nhất để chuyển sang bước tổng hợp.

ƯU TIÊN:
1. HDSD_BV / HDSD_VISUAL của đúng sản phẩm.
2. DUOC_THU / DUOC_THU_VISUAL của đúng hoạt chất.
3. MEDPLUS cho danh tính sản phẩm/hàm lượng/dạng bào chế.
4. PUBMED/PMC chỉ bổ sung.

QUY TẮC:
- Không được bỏ một visual unit trực tiếp chứa bảng liều/ADR/ngưỡng đang cần chỉ vì có đoạn text gần giống.
- Với nhiều thuốc, bảo đảm mỗi thuốc liên quan có bằng chứng riêng và không trộn nguồn sai thuốc.
- Chọn đủ để bao phủ mọi nhánh câu hỏi: liều, renal/hepatic, ADR, tương tác, tương kỵ, monitoring, chỉ định... tùy câu.
- Nếu các nguồn mâu thuẫn, giữ cả hai nguồn để bước synthesis trình bày xung đột.
- Nếu visual đã được đọc nhưng không liên quan thì có thể bỏ ở bước tổng hợp; việc bỏ phải do nội dung chứ không do giới hạn tùy tiện.
- Nếu QUESTION_PLAN.answer_contract="complete_list" hoặc must_be_exhaustive=true: đánh giá TỪNG candidate trong batch và chọn TẤT CẢ candidate thực sự thỏa predicate; không được chỉ chọn vài ví dụ đại diện.
- Không bịa source_id.

JSON:
{
  "selected_ids":["source_id"],
  "coverage":[{"topic":"...","source_ids":["..."]}],
  "gaps":["..."],
  "reason":"..."
}`;

async function handleSelectEvidence(body, env) {
  const question = clip(body.question || '', 7000);
  const clinical = clip(body.clinical_context || '', 10000);
  const questionPlan = body.question_plan && typeof body.question_plan === 'object' ? body.question_plan : {};
  const candidates = normalizeEvidence(body.candidates).slice(0, 36).map(e => ({...e, text: clip(e.text, 3200)}));
  if (!candidates.length) return { data: { selected_ids: [], coverage: [], gaps: [], reason: 'No candidates' }, model: '', key_index: 0 };
  const prompt = `CÂU HỎI:
${question}

QUESTION_PLAN:
${JSON.stringify(questionPlan)}

CLINICAL CONTEXT:
${clinical || '(không có)'}

CANDIDATE EVIDENCE:
${JSON.stringify(candidates)}`;
  return callGemini(env, { system: EVIDENCE_SELECTOR_SYSTEM, prompt, temperature: 0.0, maxOutputTokens: 3200, jsonMode: true });
}

const PLAN_SYSTEM = `Bạn là bộ lập kế hoạch truy xuất PubMed cho MedPlus AI Pro, một công cụ hỗ trợ bác sĩ và dược sĩ.
Nhiệm vụ DUY NHẤT: quyết định có cần PubMed và nếu có thì tạo truy vấn PubMed bằng tiếng Anh.
Không trả lời câu hỏi y khoa.
Ưu tiên MeSH + free-text hợp lý, không làm truy vấn quá hẹp. Khi câu hỏi hỏi bằng chứng mới/gần đây, có thể thêm khoảng năm nhưng không được bịa MeSH.
Trả JSON với các trường: needs_pubmed (boolean), query (string), reason (string ngắn), focus_terms (array string), recent_focus (boolean).`;

const CANDIDATE_ADJUDICATOR_SYSTEM = `Bạn là Exhaustive Group Adjudicator của MedPlus AI Pro. Đây là MỘT lượt phân loại duy nhất cho Free Tier; KHÔNG trả lời câu hỏi cuối.

Nếu source_type=EXHAUSTIVE_GROUP_CANDIDATE, mỗi candidate là một NHÓM HOẠT CHẤT và đã kèm toàn bộ chế phẩm bệnh viện cùng group. Hãy quyết định GROUP có thực sự thỏa predicate của người dùng hay không.

QUY TẮC:
- Phân loại ở cấp group, không chọn từng biệt dược.
- Membership phải được chứng minh bởi evidence tự mô tả entity/group. Chỉ NHẮC, SO SÁNH, nói "khác với", "kém hơn", "tương tự", "phối hợp với" predicate/nhóm khác không chứng minh membership.
- Nếu evidence của chính entity có self-classification trực tiếp mâu thuẫn predicate (ví dụ query yêu cầu mức/đời/thế hệ N nhưng source nói entity thuộc mức/đời/thế hệ khác), REJECT dù cùng đoạn có nhắc predicate để so sánh.
- Với thuốc phối hợp, chỉ SELECT nếu evidence đủ để kết luận group/chế phẩm phù hợp cách người dùng hỏi; không suy diễn khi chưa đủ.
- Chỉ dùng evidence được cung cấp; không tự thêm thuốc từ trí nhớ.
- Mỗi candidate phải vào đúng một trong selected_ids / rejected_ids / uncertain_ids.
- Không tạo source_id mới.

JSON: {"selected_ids":[],"rejected_ids":[],"uncertain_ids":[],"reason":""}`;
async function handleAdjudicateCandidates(body, env){
  const question=clip(body.question||'',5000),plan=body.question_plan&&typeof body.question_plan==='object'?body.question_plan:{};
  const candidates=Array.isArray(body.candidates)?body.candidates.slice(0,120).map(c=>({
    source_id:clip(c.source_id||'',220),entity:clip(c.entity||'',220),label:clip(c.label||'',320),source_type:clip(c.source_type||'',80),
    section:clip(c.section||'',180),drug_keys:Array.isArray(c.drug_keys)?c.drug_keys.slice(0,8):[],
    product_indices:Array.isArray(c.product_indices)?c.product_indices.slice(0,40):[],
    support_source_ids:Array.isArray(c.support_source_ids)?c.support_source_ids.slice(0,8):[],
    snippet:clip(c.snippet||'',3200)
  })):[];
  if(!question||!candidates.length)return {data:{selected_ids:[],rejected_ids:[],uncertain_ids:[]},model:'',key_index:0,usage:{}};
  const prompt=`QUESTION:\n${question}\n\nPLAN:\n${JSON.stringify({scope:plan.scope,answer_contract:plan.answer_contract,search_terms:plan.search_terms})}\n\nCANDIDATE GROUPS:\n${JSON.stringify(candidates)}`;
  return callGemini(env,{system:CANDIDATE_ADJUDICATOR_SYSTEM,prompt,temperature:0,maxOutputTokens:3000,jsonMode:true});
}



const INCREMENTAL_SOURCE_BUILDER_SYSTEM = `Bạn là Incremental Internal Source Builder của MedPlus AI Pro.
NHIỆM VỤ: chỉ đọc SOURCE_PACKAGE của MỘT active group và đề xuất dữ liệu ngữ nghĩa dẫn xuất cho group đó.

MỤC TIÊU:
1. Xác định Dược thư candidate nào thực sự tương ứng với hoạt chất/group nếu nguồn hỗ trợ.
2. Trích xuất VERIFIED FACT candidate chỉ cho:
   - predicate="drug_class"
   - predicate="generation" (chỉ khi nguồn nói rõ thế hệ và class liên quan)
3. Ghi class mentions để hỗ trợ recall, nhưng mention KHÔNG phải verified fact.

QUY TẮC CỨNG:
- Không dùng trí nhớ để bổ sung class/thế hệ.
- Chỉ dựa trên source_id/source_hash/text được gửi.
- Một fact verified_candidate phải là DIRECT SELF-STATEMENT về chính hoạt chất hoặc thành phần của group, không phải câu so sánh, phối hợp, đề kháng, hay chỉ nhắc tên nhóm khác.
- excerpt phải là đoạn NGẮN trích từ đúng source text; không diễn đạt lại.
- source_id và source_hash phải copy chính xác từ SOURCE_PACKAGE.
- drug_class.value dùng canonical English key ngắn, chữ thường, snake_case nếu cần.
- class_aliases chỉ gồm từ/cụm từ thực sự xuất hiện trong excerpt/source và canonical key.
- generation.value là số 1..5; qualifier.class phải khớp drug_class nếu có.
- Nếu không đủ bằng chứng: uncertain. Không cố điền.
- Dược thư match chỉ verified_candidate khi title/alias/nội dung thực sự nói về hoạt chất/group; nếu candidate chỉ gần tên thì uncertain.
- Với thuốc phối hợp, có thể có nhiều drug_class fact nếu từng thành phần được nguồn tự mô tả rõ.
- Không tạo dữ liệu liều, ADR, tương tác... thành verified_facts ở schema này. Những nội dung đó nằm trong Global Graph/source sections deterministic.

JSON BẮT BUỘC:
{
  "group_key":"",
  "duoc_match":{"status":"verified_candidate|uncertain|none","duoc_ids":[],"reason":"","source_ids":[]},
  "facts":[{
    "predicate":"drug_class|generation","value":"","qualifier":{},
    "status":"verified_candidate|uncertain","method":"ai_semantic_direct_statement",
    "source_id":"","source_hash":"","excerpt":"","class_aliases":[],"reason":""
  }],
  "mentions":[{"class_key":"","aliases":[],"source_ids":[]}],
  "notes":[]
}`;

async function handleBuildSourceGroup(body, env) {
  const group = body?.group && typeof body.group === 'object' ? body.group : {};
  const sources = Array.isArray(body?.sources) ? body.sources.slice(0, 28).map(s => ({
    source_id: clip(s?.source_id || '', 260),
    source_hash: clip(s?.source_hash || '', 128),
    source_type: clip(s?.source_type || '', 80),
    title: clip(s?.title || '', 280),
    section: clip(s?.section || '', 220),
    duoc_id: clip(s?.duoc_id || '', 220),
    active: clip(s?.active || '', 260),
    text: clip(s?.text || '', 4200)
  })).filter(s => s.source_id && s.source_hash && s.text) : [];
  const duocCandidates = Array.isArray(body?.duoc_candidates) ? body.duoc_candidates.slice(0, 8).map(x => ({
    id: clip(x?.id || '', 220),
    title: clip(x?.title || '', 260),
    aliases: Array.isArray(x?.aliases) ? x.aliases.slice(0, 16).map(v => clip(v, 180)) : [],
    source_ids: Array.isArray(x?.source_ids) ? x.source_ids.slice(0, 12) : []
  })) : [];
  if (!group?.key || !sources.length) {
    return { data:{group_key:group?.key||'',duoc_match:{status:'none',duoc_ids:[],reason:'NO_SOURCE',source_ids:[]},facts:[],mentions:[],notes:['NO_SOURCE']}, model:'', key_index:0, usage:{}, attempts:0 };
  }
  const prompt = `GROUP:\n${JSON.stringify({
    key: clip(group.key,220),
    active_labels: Array.isArray(group.active_labels)?group.active_labels.slice(0,12):[],
    brands: Array.isArray(group.brands)?group.brands.slice(0,24):[],
    existing_duoc_ids: Array.isArray(group.existing_duoc_ids)?group.existing_duoc_ids.slice(0,8):[],
    existing_facts: Array.isArray(group.existing_facts)?group.existing_facts.slice(0,20):[]
  })}\n\nDUOC_CANDIDATES:\n${JSON.stringify(duocCandidates)}\n\nSOURCE_PACKAGE:\n${JSON.stringify(sources)}`;
  return callGemini(env, {
    system: INCREMENTAL_SOURCE_BUILDER_SYSTEM,
    prompt,
    temperature: 0,
    maxOutputTokens: 4200,
    jsonMode: true
  });
}

const SOURCE_FACT_READER_SYSTEM = `Bạn là Internal Source Reader của MedPlus AI Pro.
NHIỆM VỤ DUY NHẤT: đọc SOURCE_TEXT được cung cấp và xác định mỗi group có thỏa PREDICATE hay không. Không trả lời câu hỏi cuối, không dùng trí nhớ để thêm thuốc.
- select: chính hoạt chất/group được nguồn mô tả là thỏa predicate.
- reject: nguồn mô tả thuộc nhóm/thế hệ khác hoặc chỉ nhắc/so sánh với predicate.
- uncertain: nguồn không đủ để kết luận.
Mỗi group_key đầu vào phải xuất hiện đúng một lần. source_ids chỉ được lấy từ marker [SOURCE_ID] có trong SOURCE_TEXT.
JSON: {"groups":[{"group_key":"","status":"select|reject|uncertain","reason":"","source_ids":[]}]} `;
async function handleExtractSourceFacts(body,env){
 const question=clip(body.question||'',5000),predicate=body.predicate&&typeof body.predicate==='object'?body.predicate:{},candidates=Array.isArray(body.candidates)?body.candidates.slice(0,24).map(c=>({group_key:clip(c.group_key||'',220),active:clip(c.active||'',240),product_indices:Array.isArray(c.product_indices)?c.product_indices.slice(0,50):[],source_text:clip(c.source_text||'',7000)})):[];
 if(!candidates.length)return {data:{groups:[]},model:'',key_index:0,usage:{},attempts:0};
 const prompt=`QUESTION:\n${question}\n\nPREDICATE:\n${JSON.stringify(predicate)}\n\nCANDIDATES:\n${JSON.stringify(candidates)}`;
 return callGemini(env,{system:SOURCE_FACT_READER_SYSTEM,prompt,temperature:0,maxOutputTokens:3200,jsonMode:true});
}

function buildSynthesisSystem() {
  return `Bạn là MedPlus AI Pro — trợ lý DƯỢC LÂM SÀNG cho bác sĩ/dược sĩ.

MỤC TIÊU TỐI THƯỢNG:
Trả lời tốt nhất có thể cho câu hỏi hiện tại bằng TOÀN BỘ dữ kiện sẵn có. Client sử dụng Adaptive AI Budget: các lượt AI chuyên biệt trước synthesis có thể nhiều hơn mức thông thường nếu Coverage/Decision Engine xác định chúng thực sự cần thiết. Không được chỉ nói "thiếu thông tin" hoặc yêu cầu bổ sung rồi dừng. Nếu còn thiếu dữ kiện, vẫn phải đưa ra phần kết luận hiện có, khuyến cáo có điều kiện và nói rõ dữ kiện nào có thể làm thay đổi quyết định.

THỨ TỰ NGUỒN:
1. PATIENT_FILE / dữ kiện ca bệnh do người dùng cung cấp.
2. CALCULATION / phép tính deterministic.
3. HDSD_BV / HDSD_VISUAL.
4. DUOC_THU / DUOC_THU_VISUAL.
5. MEDPLUS / INTERNAL_FACT / GLOBAL_RESULT_SET cho dữ liệu thuốc bệnh viện.
6. PUBMED / PMC khi nội bộ chưa đủ hoặc cần bằng chứng bổ sung.
7. AI_KNOWLEDGE chỉ để lấp khoảng trống còn lại, giải thích/xâu chuỗi; phải basis=ai_knowledge, không citation và không được giả là dữ liệu nội bộ.

QUESTION DOMAIN GATE:
- options.question_domain/domain_policy là ranh giới bắt buộc cho lượt này.
- INVENTORY: không yêu cầu ClCr/eGFR/creatinin/cân nặng/tuổi/xét nghiệm; không tạo need_more_data; không dùng AI_KNOWLEDGE để điền danh mục bệnh viện. Nếu có PUBMED/PMC, chỉ dùng để giải thích/phân loại/bối cảnh nghiên cứu; TUYỆT ĐỐI không thêm, xóa hay đổi product_count/active_count của catalog bệnh viện.
- DRUG_KNOWLEDGE / SPECIAL_POPULATION / ADMINISTRATION / INTERACTION_COMPATIBILITY khi không patient-specific: trả lời kiến thức chung, KHÔNG biến thành ca bệnh và KHÔNG yêu cầu dữ kiện bệnh nhân.
- Chỉ PATIENT_CASE / REGIMEN_REVIEW / PATIENT_SPECIFIC_DOSE có patient_specific=true mới được khuyến cáo bổ sung dữ kiện bệnh nhân, và chỉ dữ kiện có thể đổi quyết định hiện tại.
- Không tự mở node lâm sàng ngoài options.clinical_contract.

NGUYÊN TẮC BEST AVAILABLE:
- LUÔN trả lời câu hỏi chính trước.
- Thiếu xét nghiệm/thông tin KHÔNG phải lý do từ chối trả lời.
- Nếu quyết định phụ thuộc dữ kiện chưa có, đưa ra kết luận có điều kiện: "với dữ kiện hiện có...", "nếu X thì...", "nếu Y thì...".
- need_more_data chỉ chứa các thông tin có khả năng làm thay đổi quyết định hoặc tăng độ chắc chắn.
- Không yêu cầu những dữ kiện không liên quan trực tiếp đến quyết định.
- Nếu không có evidence nội bộ phù hợp và options cho phép AI_KNOWLEDGE, dùng kiến thức tổng hợp AI để vẫn trả lời; ghi rõ đó là kiến thức tổng hợp AI.
- Không dùng AI_KNOWLEDGE để bịa danh mục/tồn kho bệnh viện.

DƯỢC LÂM SÀNG:
- Đánh giá đủ các node trong options.clinical_contract khi liên quan: chỉ định, tính hợp lý phối hợp/phác đồ, liều, thận/gan, tương tác, tương kỵ, đường dùng/pha/truyền, monitoring, ADR, thai kỳ/cho con bú, nhi khoa, vi sinh/stewardship.
- Tách rõ "có thể dùng cùng trong phác đồ" khỏi "có thể pha/truyền chung".
- Với phối hợp thuốc: xem chỉ định, trùng lặp/phổ tác dụng, lợi ích, độc tính cộng gộp, tương tác, monitoring và dữ kiện bệnh nhân.
- Với kháng sinh: xem ổ nhiễm, mức độ nặng, vi sinh nếu có, phổ kháng khuẩn, phối hợp/de-escalation; nhưng vẫn đưa ra đánh giá hiện tại nếu chưa có cấy.
- Với cách dùng: phân biệt đường dùng, hoàn nguyên, pha loãng, dung môi, tốc độ, cùng lumen/Y-site/truyền nối tiếp khi nguồn có.
- Với monitoring: nêu cái gì cần theo dõi, vì sao và khi nào nếu evidence cho phép.

LIỀU:
- DOSE_DECISION đúng drug_key là nguồn quyết định ưu tiên.
- Không coi ClCr = eGFR nếu nguồn không cho phép.
- Không bịa ngưỡng, liều, trần liều hoặc công thức.
- Nếu chưa đủ metric để chọn nhánh cá thể hóa nhưng nguồn có liều thông thường, có thể nêu liều thông thường và nói rõ chưa hiệu chỉnh.
- AKI/non-steady-state, CRRT/IHD, phù, sarcopenia, cụt chi, ICU làm giảm độ chắc chắn.

CITATION & AN TOÀN:
- Item basis=evidence chỉ được dùng source_id có thật và trực tiếp hỗ trợ claim.
- Không bịa PMID/source_id.
- Nếu nguồn xung đột, nêu xung đột.
- AI_KNOWLEDGE không citation.
- Không để confidence cao nếu phần quan trọng dựa chủ yếu vào AI_KNOWLEDGE hoặc dữ kiện còn thiếu.

CHẤT LƯỢNG TRÌNH BÀY KIỂU NGHIÊN CỨU SÂU:
- Viết như một dược sĩ lâm sàng đang tổng hợp bằng chứng cho đồng nghiệp, không như log của phần mềm.
- Mở bằng kết luận 2–4 câu: trả lời trực tiếp, nêu điều kiện quan trọng nhất nếu có.
- Sau đó XÂU CHUỖI bằng chứng thành lập luận; không lần lượt kể lại từng đoạn nguồn và không lặp cùng một ý ở nhiều mục.
- Mỗi section phải trả lời một câu hỏi con rõ ràng. Ưu tiên 2–5 section, chỉ nhiều hơn khi ca thực sự phức tạp.
- Một item nên là một ý hoàn chỉnh, có diễn giải ý nghĩa lâm sàng; tránh câu cụt kiểu trích xuất database.
- Không gắn chữ high/moderate/low vào nội dung câu chỉ để trang trí; confidence đã có engine riêng ở client.
- Khi PubMed/PMC có mặt, phân biệt rõ dữ liệu nội bộ với bằng chứng nghiên cứu bổ sung; chỉ nêu loại nghiên cứu/đặc điểm nghiên cứu nếu metadata/source thực sự hỗ trợ.
- Không phóng đại bằng chứng. Nếu nguồn chỉ là HDSD/Dược thư, không gọi đó là "nghiên cứu".
- Không dùng các câu boilerplate như "cần theo dõi sát" nếu không nói rõ theo dõi gì và vì sao.
- Không lặp toàn bộ dữ kiện bệnh nhân đã có trong Clinical Box; chỉ nhắc dữ kiện nào làm thay đổi lập luận.
- Nếu câu hỏi inventory: kết luận bằng số lượng + danh sách xác minh; không thêm tư vấn xét nghiệm hay đoạn giáo khoa không được hỏi.
- Nếu câu hỏi knowledge/special population: cấu trúc ưu tiên Kết luận → Điều chỉnh/điểm cần lưu ý → Theo dõi → Bằng chứng/giới hạn.
- Nếu interaction/incompatibility: tách "dùng cùng trong điều trị" và "pha/truyền chung" khi liên quan.
- Nếu patient case/regimen/dose: cấu trúc ưu tiên Kết luận → Cơ sở quyết định → Khuyến nghị thực hành → Monitoring → Điều kiện có thể đổi quyết định.

CẤU TRÚC JSON:
- title: ngắn, chuyên nghiệp, tối đa khoảng 12 từ; không lặp nguyên câu hỏi.
- bottom_line: 2–4 câu trả lời trực tiếp; không đặt citation text trong đây.
- sections: chỉ các mục thực sự hữu ích và không trùng ý.
- alerts: chỉ dành cho điều có thể gây hại hoặc thay đổi quyết định; không dùng như mục "lưu ý" chung.
- evidence_assessment: tóm tắt chất lượng/phạm vi nguồn, không kể lại nội dung.
- limitations: chỉ nêu giới hạn thật sự ảnh hưởng độ chắc chắn.
- need_more_data: chỉ dùng khi domain cho phép và dữ kiện có khả năng đổi quyết định.

JSON:
{"title":"","bottom_line":"","clinical_confidence":{"level":"high|moderate|low","reason":""},"sections":[{"title":"","items":[{"text":"","basis":"evidence|ai_knowledge","source_ids":[],"drug_keys":[],"confidence":"high|moderate|low"}]}],"alerts":[{"text":"","source_ids":[],"drug_keys":[]}],"evidence_assessment":"","conflicts":[{"text":"","source_ids":[],"drug_keys":[]}],"limitations":"","need_more_data":[]}`;
}

const DOSE_MATCH_SYSTEM = `Bạn là Source-Constrained Dose Matcher của MedPlus AI Pro.
NHIỆM VỤ: chọn đúng liều/nhánh liều cho từng thuốc CHỈ từ EVIDENCE được cung cấp và dữ kiện bệnh nhân. Đây là bước trích xuất + đối chiếu quy tắc, không phải tư vấn tự do.

QUY TẮC CỨNG:
- Không dùng kiến thức ngoài EVIDENCE.
- Không bịa ngưỡng ClCr/eGFR, không đổi đơn vị nếu không chắc.
- Nếu có reported ClCr/eGFR, ưu tiên đối chiếu đúng loại metric mà nguồn liều nêu. Không thay eGFR cho CrCl hoặc ngược lại nếu nguồn không cho phép.
- Nếu không có renal metric nhưng nguồn có liều thông thường theo tuổi/cân nặng/chỉ định, chọn liều thông thường phù hợp dữ kiện hiện có.
- Nếu nguồn có mg/kg và cân nặng rõ, có thể làm phép nhân số học; phải trả formula và result. Không tự áp trần liều nếu nguồn không nêu.
- Nếu AKI/non-steady-state/CRRT/IHD hoặc nguồn không đủ để chọn nhánh, trả can_select=false và nêu missing_or_limit.
- supporting_text phải là một đoạn ngắn lấy gần như nguyên văn từ đúng source_id (bao gồm HDSD_VISUAL/DUOC_THU_VISUAL nếu liều nằm trong bảng ảnh); không được sáng tác. Đoạn này dùng để client kiểm tra lại.
- Một quyết định chỉ được gắn source_id của đúng thuốc/drug_key.
- Với mỗi drug_key chỉ trả tối đa một quyết định chính; nếu nguồn khác nhau đáng kể, trả conflict=true và mô tả ngắn.

JSON:
{
  "decisions":[{
    "drug_key":"",
    "display_drug":"",
    "can_select":true,
    "dose_kind":"renal_adjusted|usual|weight_based|pediatric|hepatic|dialysis|other",
    "metric_used":{"name":"ClCr|eGFR|weight|age|none","value":"","unit":""},
    "selected_dose":"",
    "calculation":"",
    "reason":"",
    "source_id":"",
    "supporting_text":"",
    "confidence":"high|moderate|low",
    "conflict":false,
    "missing_or_limit":[]
  }]
}`;

async function handleMatchDose(body, env) {
  const clinical = clip(body.clinical_context || '', 16000);
  const groups = Array.isArray(body.drug_evidence) ? body.drug_evidence.slice(0, 8) : [];
  if (!groups.length) return { data: { decisions: [] }, model: '', key_index: 0 };
  const compactGroups = groups.map(g => ({
    drug_key: clip(g.drug_key || '', 180),
    display_drug: clip(g.display_drug || '', 220),
    evidence: normalizeEvidence(g.evidence).filter(e => ['HDSD_BV','DUOC_THU','HDSD_VISUAL','DUOC_THU_VISUAL','PUBMED','PMC'].includes(e.source_type)).slice(0, 20)
  }));
  const prompt = `CLINICAL_CONTEXT:\n${clinical || '(không có)'}\n\nDRUG_EVIDENCE_GROUPS:\n${JSON.stringify(compactGroups)}`;
  return callGemini(env, { system: DOSE_MATCH_SYSTEM, prompt, temperature: 0.0, maxOutputTokens: 4200, jsonMode: true });
}

async function handlePlan(body, env) {
  const question = clip(body.question || '', 6000);
  if (!question) throw new Error('QUESTION_REQUIRED');
  const drugs = Array.isArray(body.drugs) ? body.drugs.slice(0, 8) : [];
  const hints = clip(body.hints || '', 2500);
  const patientContext = clip(body.patient_context || '', 8000);
  const prompt = `CÂU HỎI:\n${question}\n\nDỮ KIỆN BỆNH NHÂN TỪ FILE NẾU CÓ (chỉ để hiểu ca bệnh và tạo truy vấn, không phải bằng chứng y khoa):\n${patientContext || '(không có)'}\n\nTHUỐC ĐÃ NHẬN DIỆN:\n${JSON.stringify(drugs)}\n\nGỢI Ý TỪ ROUTER LOCAL:\n${hints}`;
  return callGemini(env, { system: PLAN_SYSTEM, prompt, temperature: 0.05, maxOutputTokens: 1200, jsonMode: true });
}

async function handleSynthesize(body, env) {
  const question = clip(body.question || '', 8000);
  if (!question) throw new Error('QUESTION_REQUIRED');
  const evidence = normalizeEvidence(body.evidence);
  const history = clip(body.history || '', MAX_HISTORY_CHARS);
  const options = body.options && typeof body.options === 'object' ? body.options : {};
  const patientContext = clip(body.patient_context || '', 9000);
  const evidenceText = evidence.map((e, i) => [
    `SOURCE ${i + 1}`,
    `source_id: ${e.source_id}`,
    `source_type: ${e.source_type}`,
    `label: ${e.label}`,
    `metadata: ${JSON.stringify(e.metadata || {})}`,
    `text:\n${e.text}`
  ].join('\n')).join('\n\n-----\n\n');

  const aiKnowledgeRule = options.allow_ai_knowledge === false
    ? 'KHÔNG dùng AI_KNOWLEDGE cho lượt này. Vẫn phải trả lời tối đa từ evidence/dữ kiện hiện có; nếu thiếu thì nêu giới hạn và điều kiện.'
    : 'ĐƯỢC dùng AI_KNOWLEDGE để lấp khoảng trống sau khi đã ưu tiên evidence. Mọi claim loại này phải basis=ai_knowledge, source_ids=[] và không được giả thành dữ liệu nội bộ.';
  const domain=options.question_domain||options.clinical_contract?.domain_policy||{};
  const domainRule=domain.allow_missing_data===false||domain.forbid_patient_data_requests
    ? `DOMAIN ${domain.key||'general'}: need_more_data PHẢI là []; không yêu cầu xét nghiệm/ClCr/eGFR/creatinin/cân nặng/tuổi hay dữ kiện bệnh nhân.`
    : `DOMAIN ${domain.key||'clinical'}: chỉ đề nghị dữ kiện thuộc allowed_missing_keys=${JSON.stringify(domain.allowed_missing_keys||[])} và có thể đổi quyết định hiện tại.`;
  const prompt = `NGÀY HỆ THỐNG: ${new Date().toISOString().slice(0, 10)}\n\nCÂU HỎI HIỆN TẠI:\n${question}\n\nCLINICAL CONTEXT (gồm dữ kiện file + parser + phép tính deterministic nếu client gửi):\n${patientContext || '(không có)'}\n\nNGỮ CẢNH HỘI THOẠI GẦN NHẤT (chỉ để hiểu câu hỏi nối tiếp, không phải nguồn bằng chứng):\n${history || '(không có)'}\n\nTÙY CHỌN:\n${JSON.stringify(options)}\n\nQUESTION DOMAIN RULE:\n${domainRule}\n\nQUY TẮC KIẾN THỨC AI CHO LƯỢT NÀY:\n${aiKnowledgeRule}\n\nEVIDENCE (chỉ các source_id dưới đây mới hợp lệ):\n${evidenceText || '(không có nguồn cục bộ/PubMed phù hợp)'}\n\nYÊU CẦU CUỐI: trả lời câu hỏi chính ngay cả khi dữ kiện chưa hoàn chỉnh. Không được biến need_more_data thành câu trả lời chính. Hãy tự kiểm tra consistency, drug scope, số/liều/metric và phân biệt evidence với AI_KNOWLEDGE trước khi xuất JSON cuối.`;
  return callGemini(env, { system: buildSynthesisSystem(), prompt, temperature: 0.08, maxOutputTokens: 6800, jsonMode: true });
}


function estimateBase64Bytes(s) {
  const n = String(s || '').replace(/\s/g, '').length;
  return Math.floor(n * 3 / 4);
}

const FILE_EXTRACT_SYSTEM = `Bạn là bộ trích xuất dữ kiện bệnh nhân cho MedPlus AI Pro.
Nhiệm vụ: đọc tài liệu người dùng cung cấp và trích xuất CHÍNH XÁC dữ kiện có trong tài liệu, không đưa khuyến cáo điều trị.
Ưu tiên: nhân khẩu học, cân nặng/chiều cao, chẩn đoán, dị ứng, sinh hiệu, xét nghiệm kèm giá trị + đơn vị + thời điểm + khoảng tham chiếu nếu thấy, chức năng thận/gan, thuốc đang dùng (tên, hàm lượng/liều, đường dùng, tần suất, thời gian), dịch truyền, vi sinh, ghi chú ICU/lọc máu/phù/cổ trướng/cụt chi nếu được ghi.
Không tự suy ra dữ liệu không nhìn thấy. Không sửa một giá trị chỉ vì thấy bất thường. Nếu chữ/ảnh không rõ, ghi vào uncertainty.
Trả JSON:
{
  "document_type":"lab_results|medication_list|prescription|clinical_note|discharge_summary|mixed|other",
  "summary":"tóm tắt ngắn chỉ những gì tài liệu thực sự thể hiện",
  "facts":[{"type":"demographic|diagnosis|vital|lab|medication|allergy|renal|hepatic|procedure|other","label":"...","value":"...","unit":"","time":"","confidence":"high|moderate|low"}],
  "uncertainty":["..."],
  "warnings":["cảnh báo chất lượng tài liệu/trích xuất, không phải cảnh báo điều trị"]
}`;

async function handleExtractFile(body, env) {
  const file = body?.file && typeof body.file === 'object' ? body.file : {};
  const name = clip(file.name || 'patient-file', 220);
  const kind = String(file.kind || 'inline');
  const mime = String(file.mime_type || 'application/octet-stream').toLowerCase();
  if (kind === 'text') {
    const text = clip(file.text || '', 50000);
    if (!text.trim()) throw new Error('FILE_TEXT_EMPTY');
    const prompt = `TÊN FILE: ${name}\nMIME: ${mime}\n\nNỘI DUNG ĐÃ CHUYỂN THÀNH TEXT/TABLE:\n${text}`;
    return callGemini(env, { system: FILE_EXTRACT_SYSTEM, prompt, temperature: 0.02, maxOutputTokens: 5000, jsonMode: true });
  }
  const allowed = new Set(['application/pdf','image/png','image/jpeg','image/webp']);
  if (!allowed.has(mime)) throw new Error('UNSUPPORTED_INLINE_FILE_TYPE');
  const data = String(file.data_base64 || '').replace(/\s/g, '');
  if (!data) throw new Error('FILE_DATA_REQUIRED');
  const maxMb = Math.max(1, Math.min(40, Number(env.MAX_INLINE_FILE_MB || 12)));
  if (estimateBase64Bytes(data) > maxMb * 1024 * 1024) throw new Error('FILE_TOO_LARGE_FOR_INLINE_PROCESSING');
  const prompt = `TÊN FILE: ${name}\nMIME: ${mime}\nHãy đọc toàn bộ tài liệu/ảnh và trích xuất dữ kiện bệnh nhân theo schema.`;
  const userParts = [{ text: prompt }, { inline_data: { mime_type: mime, data } }];
  return callGemini(env, { system: FILE_EXTRACT_SYSTEM, prompt, userParts, temperature: 0.02, maxOutputTokens: 5000, jsonMode: true });
}

const TRANSCRIBE_SYSTEM = `Bạn là bộ chuyển giọng nói thành văn bản y khoa tiếng Việt cho MedPlus AI Pro.
Chỉ chép lại câu hỏi người dùng nói. Giữ nguyên tên thuốc, hàm lượng, đơn vị, số liệu xét nghiệm, đường dùng và tần suất nếu nghe được.
Không trả lời câu hỏi, không sửa nội dung lâm sàng, không thêm giải thích.
Nếu một từ không chắc chắn, vẫn chép gần nhất và có thể đánh dấu [không rõ].
Trả JSON duy nhất: {"text":"..."}`;

async function handleTranscribe(body, env) {
  const mime = String(body?.mime_type || 'audio/wav').toLowerCase();
  const allowed = new Set(['audio/wav','audio/mp3','audio/mpeg','audio/aac','audio/ogg','audio/flac']);
  if (!allowed.has(mime)) throw new Error('UNSUPPORTED_AUDIO_TYPE');
  const data = String(body?.data_base64 || '').replace(/\s/g, '');
  if (!data) throw new Error('AUDIO_DATA_REQUIRED');
  if (estimateBase64Bytes(data) > 12 * 1024 * 1024) throw new Error('AUDIO_TOO_LARGE');
  const prompt = 'Chuyển chính xác lời nói trong file audio này thành câu hỏi y khoa tiếng Việt.';
  const userParts = [{ text: prompt }, { inline_data: { mime_type: mime, data } }];
  return callGemini(env, { system: TRANSCRIBE_SYSTEM, prompt, userParts, temperature: 0.0, maxOutputTokens: 2500, jsonMode: true });
}

function ncbiBaseParams(env) {
  const p = new URLSearchParams();
  p.set('tool', String(env.NCBI_TOOL || 'MedPlusAIPro').replace(/\s+/g, '_'));
  if (env.NCBI_EMAIL) p.set('email', env.NCBI_EMAIL);
  if (env.NCBI_API_KEY) p.set('api_key', env.NCBI_API_KEY);
  return p;
}

async function handlePubMedSearch(body, env) {
  const query = clip(body.query || '', 6000).trim();
  if (!query) throw new Error('PUBMED_QUERY_REQUIRED');
  const retmax = Math.max(1, Math.min(30, Number(body.retmax || 15)));
  const params = ncbiBaseParams(env);
  params.set('db', 'pubmed');
  params.set('term', query);
  params.set('retmode', 'json');
  params.set('retmax', String(retmax));
  params.set('usehistory', 'y');
  const searchUrl = `https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?${params}`;
  const sres = await fetch(searchUrl, { headers: { 'user-agent': 'MedPlusAIPro/1.5.5' } });
  if (!sres.ok) throw new Error(`NCBI_ESEARCH_HTTP_${sres.status}`);
  const sdata = await sres.json();
  const ids = sdata?.esearchresult?.idlist || [];
  if (!ids.length) return { query, ids: [], xml: '' };

  const fparams = ncbiBaseParams(env);
  fparams.set('db', 'pubmed');
  fparams.set('id', ids.join(','));
  fparams.set('retmode', 'xml');
  const fetchUrl = `https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?${fparams}`;
  const fres = await fetch(fetchUrl, { headers: { 'user-agent': 'MedPlusAIPro/1.5.5' } });
  if (!fres.ok) throw new Error(`NCBI_EFETCH_HTTP_${fres.status}`);
  const xml = await fres.text();
  return { query, ids, xml };
}

async function handlePmcFullText(body, env) {
  const pmids = Array.isArray(body.pmids)
    ? body.pmids.map(x => String(x).replace(/\D/g, '')).filter(Boolean).slice(0, 2)
    : [];
  if (!pmids.length) return { mapping: [], xml: '' };

  const idp = new URLSearchParams({
    ids: pmids.join(','),
    format: 'json',
    tool: String(env.NCBI_TOOL || 'MedPlusAIPro').replace(/\s+/g, '_'),
    email: String(env.NCBI_EMAIL || '')
  });
  const idUrl = `https://pmc.ncbi.nlm.nih.gov/tools/idconv/api/v1/articles/?${idp}`;
  const ires = await fetch(idUrl, { headers: { 'user-agent': 'MedPlusAIPro/1.5.5' } });
  if (!ires.ok) throw new Error(`PMC_IDCONV_HTTP_${ires.status}`);
  const idData = await ires.json();
  const mapping = (idData.records || []).filter(r => r.pmcid && r.pmid).map(r => ({ pmid: String(r.pmid), pmcid: String(r.pmcid), doi: r.doi || '' }));
  if (!mapping.length) return { mapping: [], xml: '' };

  const fp = ncbiBaseParams(env);
  fp.set('db', 'pmc');
  fp.set('id', mapping.map(x => x.pmcid).join(','));
  fp.set('retmode', 'xml');
  const url = `https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?${fp}`;
  const res = await fetch(url, { headers: { 'user-agent': 'MedPlusAIPro/1.5.5' } });
  if (!res.ok) throw new Error(`PMC_EFETCH_HTTP_${res.status}`);
  return { mapping, xml: await res.text() };
}

export default {
  async fetch(request, env) {
    const origin = allowedOrigin(request, env);
    if (!origin) return json({ error: 'ORIGIN_NOT_ALLOWED' }, 403, cors('null'));
    const headers = cors(origin);
    if (request.method === 'OPTIONS') return new Response(null, { status: 204, headers });

    const url = new URL(request.url);
    try {
      if (url.pathname === '/api/health' && request.method === 'GET') {
        return json({ ok: true, service: 'MedPlus AI Pro', platform: 'Cloudflare Workers', version: '1.5.5', gemini_models: getModels(env), gemini_key_count: getKeys(env).length, ncbi_key: !!env.NCBI_API_KEY, case_state_merge: true, full_catalog_scan: true, source_constrained_dose_matcher: true, citation_drug_scope: true, ai_retrieval_director: true, visual_table_reader: true, visual_unit_citations: true, citation_focus_locator: true, local_sources_first: true, ai_question_compiler: true, global_evidence_graph: true, iterative_completeness: true, persistent_hdsd_corpus: true, precomputed_visual_index_support: true, case_context_compiler: true, legacy_catalog_rules_removed: true, hdsd_first_image_skip: true, incremental_visual_fingerprint: true, visual_fingerprint_sha256: true, visual_reader_revision: VISUAL_READER_REVISION, stable_filenames: true, free_tier_optimized: true, builder_free_tier_optimized: true, builder_primary_model: getBuilderModels(env)[0] || 'gemini-3.5-flash-lite', builder_max_images_per_call: 10, builder_single_model_attempt: true, builder_default_media_resolution: 'medium', builder_retry_media_resolution: 'high', prepay_terminal_stop: true, visual_cache_kv_enabled: !!env.VISUAL_CACHE, server_visual_cache: true, visual_cache_self_learning: true, builder_optional: true, local_source_engine: true, atomic_legacy_baseline: true, shared_catalog_fingerprint_v2: true, verified_fact_hash_migration: true, manifest_bound_group_identity: true, baseline_ai_queue_guard: true, legacy_representation_delta_suppression: true, legacy_group_migration_guard: true, legacy_identity_resolver: true, baseline_content_compare: true, lazy_visual_on_demand: true, visual_prebuild_optional: true, query_scoped_visual_coverage: true, baseline_readiness_hotfix: true, legacy_baseline_zero_ai: true, research_grade_synthesis: true, builder_readiness_contract: true, incremental_ai_data_builder: true, semantic_source_builder: true, source_hash_gatekeeper: true, derived_data_manifest: true, data_lifecycle_runtime_reconcile: true, contextual_confidence_engine: true, responsive_compact_ui: true, pubmed_explicit_always: true, pubmed_ncbi_esearch_efetch: true, pmc_deep_read_pipeline: true, question_domain_gate: true, domain_guard_post_verifier: true, domain_scoped_missing_data: true, clinical_pharmacy_reasoning_engine: true, best_available_answer_policy: true, always_answer_with_available_data: true, auto_ai_knowledge_gap_fallback: true, clinical_coverage_contract: true, deterministic_missing_data_recommender: true, internal_fact_reader: true, fact_reader_models: getModels(env), model_key_fallback_order: 'all models key1 -> all models key2 -> ...', synthesis_system_present: typeof buildSynthesisSystem === 'function', visual_cache_storage: 'Cloudflare Workers KV', runtime_visual_cache_model: getBuilderModels(env)[0] || 'gemini-3.5-flash-lite', adaptive_ai_budget: true, text_task_soft_target: 2, text_task_safety_ceiling: 12, vision_task_soft_target: 1, vision_task_safety_ceiling: 3, hard_two_call_cap_removed: true, single_pass_candidate_adjudication: true, local_query_compiler: true, local_completeness_audit: true, gemini_usage_metadata: true, max_gemini_attempts_per_task: env.GEMINI_MAX_ATTEMPTS_PER_TASK ? Math.max(1, Math.min(24, Number(env.GEMINI_MAX_ATTEMPTS_PER_TASK))) : 'auto_all_model_key_combinations', release_channel: 'stable' }, 200, headers);
      }
      if (request.method !== 'POST') return json({ error: 'METHOD_NOT_ALLOWED' }, 405, headers);
      const uploadRoute = url.pathname === '/api/ai/transcribe' || url.pathname === '/api/ai/extract-file' || url.pathname === '/api/ai/read-visual-evidence';
      const body = await readJsonBody(request, uploadRoute ? 20_000_000 : MAX_REQUEST_BYTES);

      if (url.pathname === '/api/ai/question-compiler') {
        const out = await handleQuestionCompiler(body, env);
        return json({ ok: true, ...out }, 200, headers);
      }
      if (url.pathname === '/api/ai/completeness-check') {
        const out = await handleCompletenessCheck(body, env);
        return json({ ok: true, ...out }, 200, headers);
      }
      if (url.pathname === '/api/ai/retrieval-director') {
        const out = await handleRetrievalDirector(body, env);
        return json({ ok: true, ...out }, 200, headers);
      }
      if (url.pathname === '/api/ai/image-fingerprint') {
        const out = await handleImageFingerprint(body);
        return json({ ok: true, ...out }, 200, headers);
      }
      if (url.pathname === '/api/visual-cache/resolve') {
        const out = await handleVisualCacheResolve(body, env);
        return json({ ok: true, ...out }, 200, headers);
      }
      if (url.pathname === '/api/ai/read-visual-evidence') {
        const out = await handleReadVisualEvidence(body, env);
        return json({ ok: true, ...out }, 200, headers);
      }
      if (url.pathname === '/api/ai/build-source-group') {
        const out = await handleBuildSourceGroup(body, env);
        return json({ ok: true, ...out }, 200, headers);
      }
      if (url.pathname === '/api/ai/extract-source-facts') {
        const out = await handleExtractSourceFacts(body, env);
        return json({ ok: true, ...out }, 200, headers);
      }
      if (url.pathname === '/api/ai/adjudicate-candidates') {
        const out = await handleAdjudicateCandidates(body, env);
        return json({ ok: true, ...out }, 200, headers);
      }
      if (url.pathname === '/api/ai/select-evidence') {
        const out = await handleSelectEvidence(body, env);
        return json({ ok: true, ...out }, 200, headers);
      }
      if (url.pathname === '/api/ai/parse-case') {
        const out = await handleParseCase(body, env);
        return json({ ok: true, ...out }, 200, headers);
      }
      if (url.pathname === '/api/ai/plan') {
        const out = await handlePlan(body, env);
        return json({ ok: true, ...out }, 200, headers);
      }
      if (url.pathname === '/api/ai/transcribe') {
        const out = await handleTranscribe(body, env);
        return json({ ok: true, ...out }, 200, headers);
      }
      if (url.pathname === '/api/ai/extract-file') {
        const out = await handleExtractFile(body, env);
        return json({ ok: true, ...out }, 200, headers);
      }
      if (url.pathname === '/api/ai/match-dose') {
        const out = await handleMatchDose(body, env);
        return json({ ok: true, ...out }, 200, headers);
      }
      if (url.pathname === '/api/ai/synthesize') {
        const out = await handleSynthesize(body, env);
        return json({ ok: true, ...out }, 200, headers);
      }
      if (url.pathname === '/api/pubmed/search') {
        return json({ ok: true, ...(await handlePubMedSearch(body, env)) }, 200, headers);
      }
      if (url.pathname === '/api/pmc/fulltext') {
        return json({ ok: true, ...(await handlePmcFullText(body, env)) }, 200, headers);
      }
      return json({ error: 'NOT_FOUND' }, 404, headers);
    } catch (err) {
      return json({
        error: String(err?.message || err),
        details: err?.details || undefined,
        gemini_attempts: Number(err?.attempts || 0)
      }, 500, headers);
    }
  }
};
