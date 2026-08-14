/**
 * MedPlus AI Pro - Cloudflare Worker
 * Version: 0.7.0
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
const MAX_EVIDENCE_ITEMS = 60;
const MAX_EVIDENCE_CHARS = 120_000;
const MAX_HISTORY_CHARS = 6_000;

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
    const text = clip(item.text, Math.min(6000, remaining));
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

function extractGeminiText(data) {
  const parts = data?.candidates?.[0]?.content?.parts || [];
  return parts.map(p => p?.text || '').join('').trim();
}

async function callGemini(env, { system, prompt, userParts = null, temperature = 0.12, maxOutputTokens = 8192, jsonMode = true }) {
  const keys = getKeys(env);
  if (!keys.length) throw new Error('GEMINI_KEY_NOT_CONFIGURED');
  const models = getModels(env);
  const errors = [];

  // User preference from the MedPlus project: exhaust all fallback models on key 1,
  // then move to key 2, etc.
  for (let keyIndex = 0; keyIndex < keys.length; keyIndex++) {
    const key = keys[keyIndex];
    for (let modelIndex = 0; modelIndex < models.length; modelIndex++) {
      const model = models[modelIndex];
      try {
        const url = `https://generativelanguage.googleapis.com/v1beta/models/${encodeURIComponent(model)}:generateContent`;
        const payload = {
          system_instruction: { parts: [{ text: system }] },
          contents: [{ role: 'user', parts: Array.isArray(userParts) && userParts.length ? userParts : [{ text: prompt }] }],
          generationConfig: {
            temperature,
            maxOutputTokens,
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
          // 400/401/403 are usually request/auth/location errors and retrying every model/key only wastes quota.
          // 404 can be model-specific, while 408/429/5xx are transient/fallback-worthy.
          if ([400,401,403].includes(res.status)) {
            const error = new Error('GEMINI_REQUEST_REJECTED');
            error.details = errors.slice(-12);
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
        return { data: parsed, model, key_index: keyIndex + 1 };
      } catch (err) {
        if (err?.terminalGeminiRequest) throw err;
        errors.push({ key: keyIndex + 1, model, status: 0, message: String(err?.message || err) });
      }
    }
  }
  const error = new Error('ALL_GEMINI_FALLBACKS_FAILED');
  error.details = errors.slice(-12);
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
  const res = await fetch(rawUrl, { redirect: 'follow', headers: { 'user-agent': 'MedPlusAIPro-Visual/0.7' } });
  if (!res.ok) throw new Error(`VISUAL_FETCH_HTTP_${res.status}`);
  const len = Number(res.headers.get('content-length') || 0);
  if (len && len > maxBytes) throw new Error('VISUAL_URL_TOO_LARGE');
  const mime = String(res.headers.get('content-type') || '').split(';')[0].toLowerCase();
  if (!['image/png','image/jpeg','image/webp'].includes(mime)) throw new Error('VISUAL_URL_BAD_MIME');
  const buf = await res.arrayBuffer();
  if (buf.byteLength > maxBytes) throw new Error('VISUAL_URL_TOO_LARGE');
  return { mime, data: bytesToBase64Worker(new Uint8Array(buf)) };
}

async function handleReadVisualEvidence(body, env) {
  const question = clip(body.question || '', 6000);
  const clinical = clip(body.clinical_context || '', 9000);
  const images = Array.isArray(body.images) ? body.images.slice(0, 6) : [];
  if (!images.length) return { data: { items: [] }, model: '', key_index: 0 };
  const userParts = [{ text: `CÂU HỎI HIỆN TẠI (chỉ để biết ngữ cảnh; vẫn phải đọc toàn bộ ảnh):\n${question}\n\nCLINICAL CONTEXT:\n${clinical || '(không có)'}\n\nSau đây là ${images.length} ảnh. Hãy đọc đầy đủ từng ảnh.` }];
  let totalBytes = 0;
  for (let i = 0; i < images.length; i++) {
    const im = images[i] || {};
    let mime = String(im.mime_type || '').toLowerCase();
    let data = String(im.data_base64 || '').replace(/\s/g, '');
    if (!data && im.image_url) {
      const fetched = await fetchVisualUrlAsBase64(im.image_url, 5 * 1024 * 1024);
      mime = fetched.mime;
      data = fetched.data;
    }
    if (!mime) mime = 'image/png';
    if (!['image/png','image/jpeg','image/webp'].includes(mime)) throw new Error('UNSUPPORTED_VISUAL_MIME');
    if (!data) throw new Error('VISUAL_DATA_REQUIRED');
    totalBytes += estimateBase64Bytes(data);
    if (totalBytes > 14 * 1024 * 1024) throw new Error('VISUAL_BATCH_TOO_LARGE');
    userParts.push({ text: `ẢNH ${i + 1}\nsource_id: ${clip(im.source_id || '', 260)}\nsource_type: ${clip(im.source_type || '', 80)}\nlabel: ${clip(im.label || '', 500)}\nmetadata: ${clip(JSON.stringify(im.metadata || {}), 3500)}` });
    userParts.push({ inline_data: { mime_type: mime, data } });
  }
  return callGemini(env, { system: VISUAL_EVIDENCE_SYSTEM, prompt: '', userParts, temperature: 0.0, maxOutputTokens: 9000, jsonMode: true });
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

function buildSynthesisSystem() {
  return `Bạn là MedPlus AI Pro, một Clinical Drug Intelligence Assistant dành cho bác sĩ/dược sĩ.
MỤC TIÊU: trả lời trực tiếp câu hỏi sau khi AI Question Compiler, Global Evidence Graph, HDSD corpus, visual evidence, dữ kiện bệnh nhân và các bộ kiểm chứng đã hoàn tất retrieval. Không được biến top-k retrieval thành câu trả lời nếu QUESTION_PLAN/COMPLETENESS yêu cầu exhaustive. Citation là căn cứ kiểm chứng, KHÔNG phải câu trả lời thay cho phân tích.

VAI TRÒ NGUỒN:
0) PATIENT_FILE: dữ kiện người bệnh trích từ PDF/ảnh/Excel. Chỉ chứng minh dữ kiện bệnh nhân, không phải khuyến cáo y khoa.
1) CALCULATION: phép tính deterministic của Clinical Calculator Engine (BSA, BMI, eGFR, CrCl, Child-Pugh...). Chỉ chứng minh giá trị dẫn xuất/phương pháp tính; KHÔNG được dùng CALCULATION đơn độc để tạo khuyến cáo liều.
2) HDSD_BV + HDSD_VISUAL: thông tin cấp sản phẩm đang có tại bệnh viện. HDSD_VISUAL là bảng/hình đã được Gemini Vision đọc trực tiếp từ ảnh nguồn và phải được coi ngang hàng với phần chữ HDSD.
3) DUOC_THU + DUOC_THU_VISUAL: Dược thư Quốc gia Việt Nam cấp hoạt chất/chuyên luận. DUOC_THU_VISUAL là bảng/hình gốc từ Dược thư đã được Vision đọc; đặc biệt quan trọng với bảng liều, ADR, hiệu chỉnh thận/gan.
4) PUBMED/PMC: bằng chứng nghiên cứu quốc tế, chỉ BỔ SUNG sau khi dữ liệu nội bộ MedPlus/HDSD/Dược thư đã được khai thác.
5) MEDPLUS: drug master/index (biệt dược, hoạt chất, hàm lượng, dạng bào chế, link); không mặc định là bằng chứng lâm sàng.
6) CATALOG_SCAN: kết quả quét DETERMINISTIC toàn bộ danh mục bệnh viện, Dược thư classification index và HDSD. Khi câu hỏi hỏi "bệnh viện có những thuốc nào", đây là nguồn authoritative về DANH SÁCH; không được tự bỏ bớt hoặc thêm thuốc ngoài danh sách.
7) DOSE_DECISION: kết quả Source-Constrained Dose Matcher đã ghép dữ kiện bệnh nhân (tuổi/cân nặng/ClCr/eGFR...) vào đúng nhánh liều có trong HDSD/Dược thư. Ưu tiên nguồn này khi có và luôn giữ citation nguồn gốc.
8) GLOBAL_GRAPH / HDSD_CORPUS: bằng chứng/candidate đến từ quét toàn kho, dùng để bảo đảm không bỏ sót entity/section trước khi mở nguồn gốc.
9) AI_KNOWLEDGE: kiến thức tổng hợp của mô hình, chỉ dùng để giải thích/kết nối khi cần; không gắn citation giả.

CÁCH SUY LUẬN CA BỆNH:
- Trước khi kết luận, tự phân rã vấn đề: chỉ định, liều, chức năng thận/gan, tuổi/cân nặng, tương tác thuốc-thuốc, tương tác thuốc-bệnh nhân, tương kỵ pha/truyền, độc tính cộng gộp, monitoring, dữ liệu còn thiếu.
- Phân biệt rõ TƯƠNG KỴ VẬT LÝ/PHA TRUYỀN với CHỐNG CHỈ ĐỊNH hoặc TƯƠNG TÁC LÂM SÀNG. Tương kỵ pha chung không tự động có nghĩa hai thuốc không được dùng trên cùng bệnh nhân.
- Với nhiều thuốc, đánh giá các cặp liên quan và nguy cơ cộng gộp (ví dụ độc thận/độc tai/QT/chảy máu) rồi mới tổng hợp.
- Với định liều, nếu có DOSE_DECISION hợp lệ cho đúng thuốc thì phải dùng liều/nhánh đã được matcher chọn và giải thích metric nào đã được áp dụng. Nếu chưa có DOSE_DECISION, dùng giá trị từ CALCULATION để xác định bệnh nhân nằm ở nhánh/hàng nào của bảng hoặc quy tắc trong HDSD/DUOC_THU/PUBMED. Không suy ra liều chỉ từ eGFR/CrCl nếu evidence không chứa quy tắc liều phù hợp.
- Nếu bệnh nhân đã cung cấp ClCr hoặc eGFR, không được trả một liều "chung chung" khi evidence có quy tắc hiệu chỉnh phù hợp: phải chọn đúng nhánh của giá trị đó. Nếu không có ClCr/eGFR nhưng đủ tuổi/cân nặng/chỉ định để nguồn cho liều thông thường, phải đưa ra liều thông thường phù hợp dữ kiện hiện có và nói rõ chưa áp dụng hiệu chỉnh thận.
- Khi options.question_plan.must_be_exhaustive=true hoặc answer_contract=complete_list, phải phản ánh ĐẦY ĐỦ tập kết quả deterministic/global scan đã được cung cấp; không được rút gọn chỉ vì context dài.
- Nếu options.completeness.complete=false, phải nói rõ giới hạn còn thiếu thay vì tuyên bố danh sách/đánh giá là đầy đủ.
- Phương pháp tính được chính nguồn của thuốc dùng trong nghiên cứu/nhãn/bảng liều phải được ưu tiên hơn quy tắc chung. Nếu nguồn không nói rõ, trình bày bất định thay vì tự chọn một công thức với vẻ chắc chắn.
- Nếu AKI/non-steady-state, phù/cổ trướng, sarcopenia, cụt chi, ICU/ARC, CRRT/IHD hoặc kiểu hình khác làm phép ước tính kém tin cậy, phải phản ánh điều đó trong khuyến nghị và confidence; không biến một con số ước tính thành “sự thật”.
- Nếu thiếu dữ liệu, VẪN trả lời phần có thể trả lời và nói rõ điều gì chưa thể chốt; không trả kiểu “hãy xem tài liệu”.
- Nếu options.continuation=true và câu hiện tại chủ yếu là BỔ SUNG DỮ KIỆN (ví dụ chỉ thêm creatinin, chiều cao, xét nghiệm, thuốc mới), KHÔNG chỉ xác nhận “đã nhận”. Hãy dùng hồ sơ ca bệnh đã hợp nhất + HISTORY để ĐÁNH GIÁ LẠI câu hỏi/quyết định lâm sàng trước đó và nêu rõ điều gì thay đổi sau dữ kiện mới.

TRÍCH DẪN VÀ AN TOÀN:
- Mọi item basis="evidence" phải có ít nhất một source_id có thật trong EVIDENCE và nội dung source phải trực tiếp hỗ trợ luận điểm.
- ƯU TIÊN NGUỒN CỤC BỘ: nếu HDSD/Dược thư (kể cả bảng ảnh) đã trả lời trực tiếp thì phải dùng chúng làm trục chính. PubMed/PMC chỉ bổ sung, cập nhật hoặc giải quyết khoảng trống/xung đột.
- Nếu một luận điểm về liều/ADR/ngưỡng nằm trong HDSD_VISUAL hoặc DUOC_THU_VISUAL, ưu tiên citation tới đúng visual evidence unit đó vì nó mang locator/bbox để giao diện focus đúng hàng/cell nguồn.
- Mỗi item nói về một hay nhiều thuốc phải trả thêm drug_keys là mảng canonical key của các thuốc đang được nói tới (lấy từ metadata.drug_keys của evidence). Không gắn HDSD/Dược thư của thuốc A cho luận điểm về thuốc B. Với tương tác A+B, drug_keys phải chứa cả hai nếu luận điểm thực sự nói về cả hai.
- PATIENT_FILE có thể citation cho dữ kiện ca bệnh; CALCULATION có thể citation cho phép tính; khuyến cáo y khoa phải được neo vào HDSD_BV/DUOC_THU/HDSD_VISUAL/DUOC_THU_VISUAL/PUBMED/PMC hoặc được ghi rõ basis="ai_knowledge" nếu chỉ là diễn giải tổng hợp.
- Không tự tạo PMID/source_id. Chỉ dùng source_id đúng nguyên văn.
- Không bịa liều, ngưỡng, công thức, tác dụng phụ, chống chỉ định hay monitoring.
- Khi các nguồn khác nhau, trình bày khác biệt và giải thích; không hòa trộn thành đồng thuận giả.
- Khi dữ kiện OCR/trích file không chắc, không dùng nó để chốt liều nếu sai số có thể đổi quyết định.
- Confidence KHÔNG phải “mức tự tin của AI”. Nó phải phản ánh: độ đầy đủ dữ kiện bệnh nhân + độ tin cậy phép tính + mức phủ bằng chứng + mức đồng thuận nguồn + mức trực tiếp của nguồn với đúng quần thể/thuốc.
- Trả lời tiếng Việt, ưu tiên kết luận thực hành: có/không/có điều kiện; liều/phương án nếu đủ căn cứ; cách dùng/tách đường truyền nếu liên quan; monitoring; điều kiện phải đánh giá lại.

JSON BẮT BUỘC:
{
  "title": "tiêu đề ngắn",
  "bottom_line": "kết luận trực tiếp cho chính ca bệnh",
  "clinical_confidence": {"level":"high|moderate|low","reason":"lý do dựa trên dữ kiện + phép tính + bằng chứng"},
  "sections": [
    {"title":"...","items":[{"text":"...","basis":"evidence|ai_knowledge","source_ids":["ID"],"drug_keys":["canonical-active-key"],"confidence":"high|moderate|low"}]}
  ],
  "alerts": [{"text":"...","source_ids":["ID"],"drug_keys":["canonical-active-key"]}],
  "evidence_assessment": "đánh giá nguồn/bằng chứng đã dùng",
  "conflicts": [{"text":"...","source_ids":["ID1","ID2"],"drug_keys":["canonical-active-key"]}],
  "limitations": "giới hạn hiện tại",
  "need_more_data": ["dữ liệu bổ sung có thể làm thay đổi quyết định"]
}`;
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
  const patientContext = clip(body.patient_context || '', 14000);
  const evidenceText = evidence.map((e, i) => [
    `SOURCE ${i + 1}`,
    `source_id: ${e.source_id}`,
    `source_type: ${e.source_type}`,
    `label: ${e.label}`,
    `metadata: ${JSON.stringify(e.metadata || {})}`,
    `text:\n${e.text}`
  ].join('\n')).join('\n\n-----\n\n');

  const aiKnowledgeRule = options.allow_ai_knowledge === false
    ? 'KHÔNG sử dụng kiến thức AI bổ sung; không tạo item basis=ai_knowledge. Nếu evidence không đủ, chỉ nêu giới hạn.'
    : 'Có thể dùng kiến thức AI bổ sung để giải thích khi hữu ích, nhưng phải basis=ai_knowledge và không có citation.';
  const prompt = `NGÀY HỆ THỐNG: ${new Date().toISOString().slice(0, 10)}\n\nCÂU HỎI HIỆN TẠI:\n${question}\n\nCLINICAL CONTEXT (gồm dữ kiện file + parser + phép tính deterministic nếu client gửi):\n${patientContext || '(không có)'}\n\nNGỮ CẢNH HỘI THOẠI GẦN NHẤT (chỉ để hiểu câu hỏi nối tiếp, không phải nguồn bằng chứng):\n${history || '(không có)'}\n\nTÙY CHỌN:\n${JSON.stringify(options)}\n\nQUY TẮC KIẾN THỨC AI CHO LƯỢT NÀY:\n${aiKnowledgeRule}\n\nEVIDENCE (chỉ các source_id dưới đây mới hợp lệ):\n${evidenceText || '(không có nguồn cục bộ/PubMed phù hợp)'}\n\nHãy tự kiểm tra consistency trước khi xuất JSON cuối.`;
  return callGemini(env, { system: buildSynthesisSystem(), prompt, temperature: 0.10, maxOutputTokens: 8192, jsonMode: true });
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
  const sres = await fetch(searchUrl, { headers: { 'user-agent': 'MedPlusAIPro/0.7' } });
  if (!sres.ok) throw new Error(`NCBI_ESEARCH_HTTP_${sres.status}`);
  const sdata = await sres.json();
  const ids = sdata?.esearchresult?.idlist || [];
  if (!ids.length) return { query, ids: [], xml: '' };

  const fparams = ncbiBaseParams(env);
  fparams.set('db', 'pubmed');
  fparams.set('id', ids.join(','));
  fparams.set('retmode', 'xml');
  const fetchUrl = `https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?${fparams}`;
  const fres = await fetch(fetchUrl, { headers: { 'user-agent': 'MedPlusAIPro/0.7' } });
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
  const ires = await fetch(idUrl, { headers: { 'user-agent': 'MedPlusAIPro/0.7' } });
  if (!ires.ok) throw new Error(`PMC_IDCONV_HTTP_${ires.status}`);
  const idData = await ires.json();
  const mapping = (idData.records || []).filter(r => r.pmcid && r.pmid).map(r => ({ pmid: String(r.pmid), pmcid: String(r.pmcid), doi: r.doi || '' }));
  if (!mapping.length) return { mapping: [], xml: '' };

  const fp = ncbiBaseParams(env);
  fp.set('db', 'pmc');
  fp.set('id', mapping.map(x => x.pmcid).join(','));
  fp.set('retmode', 'xml');
  const url = `https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?${fp}`;
  const res = await fetch(url, { headers: { 'user-agent': 'MedPlusAIPro/0.7' } });
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
        return json({ ok: true, service: 'MedPlus AI Pro', platform: 'Cloudflare Workers', version: '0.7.0', gemini_models: getModels(env), gemini_key_count: getKeys(env).length, ncbi_key: !!env.NCBI_API_KEY, case_state_merge: true, full_catalog_scan: true, source_constrained_dose_matcher: true, citation_drug_scope: true, ai_retrieval_director: true, visual_table_reader: true, visual_unit_citations: true, citation_focus_locator: true, local_sources_first: true, ai_question_compiler: true, global_evidence_graph: true, iterative_completeness: true, persistent_hdsd_corpus: true, precomputed_visual_index_support: true, case_context_compiler: true, legacy_catalog_rules_removed: true }, 200, headers);
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
      if (url.pathname === '/api/ai/read-visual-evidence') {
        const out = await handleReadVisualEvidence(body, env);
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
        details: err?.details || undefined
      }, 500, headers);
    }
  }
};
