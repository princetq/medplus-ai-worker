/**
 * MedPlus AI Pro - Cloudflare Worker
 * Version: 0.4.0
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

const MAX_REQUEST_BYTES = 350_000;
const MAX_EVIDENCE_ITEMS = 28;
const MAX_EVIDENCE_CHARS = 80_000;
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
          errors.push({ key: keyIndex + 1, model, status: res.status, message: data?.error?.message || text.slice(0, 180) });
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

const PLAN_SYSTEM = `Bạn là bộ lập kế hoạch truy xuất PubMed cho MedPlus AI Pro, một công cụ hỗ trợ bác sĩ và dược sĩ.
Nhiệm vụ DUY NHẤT: quyết định có cần PubMed và nếu có thì tạo truy vấn PubMed bằng tiếng Anh.
Không trả lời câu hỏi y khoa.
Ưu tiên MeSH + free-text hợp lý, không làm truy vấn quá hẹp. Khi câu hỏi hỏi bằng chứng mới/gần đây, có thể thêm khoảng năm nhưng không được bịa MeSH.
Trả JSON với các trường: needs_pubmed (boolean), query (string), reason (string ngắn), focus_terms (array string), recent_focus (boolean).`;

function buildSynthesisSystem() {
  return `Bạn là MedPlus AI Pro, một Clinical Drug Intelligence Assistant dành cho bác sĩ/dược sĩ.
MỤC TIÊU: trả lời trực tiếp câu hỏi lâm sàng của ca bệnh sau khi kết nối dữ kiện bệnh nhân, phép tính deterministic và bằng chứng đã truy xuất. Citation là căn cứ kiểm chứng, KHÔNG phải câu trả lời thay cho phân tích.

VAI TRÒ NGUỒN:
0) PATIENT_FILE: dữ kiện người bệnh trích từ PDF/ảnh/Excel. Chỉ chứng minh dữ kiện bệnh nhân, không phải khuyến cáo y khoa.
1) CALCULATION: phép tính deterministic của Clinical Calculator Engine (BSA, BMI, eGFR, CrCl, Child-Pugh...). Chỉ chứng minh giá trị dẫn xuất/phương pháp tính; KHÔNG được dùng CALCULATION đơn độc để tạo khuyến cáo liều.
2) HDSD_BV: thông tin cấp sản phẩm đang có tại bệnh viện.
3) DUOC_THU: Dược thư Quốc gia Việt Nam cấp hoạt chất/chuyên luận.
4) PUBMED/PMC: bằng chứng nghiên cứu quốc tế.
5) MEDPLUS: drug master/index (biệt dược, hoạt chất, hàm lượng, dạng bào chế, link); không mặc định là bằng chứng lâm sàng.
6) AI_KNOWLEDGE: kiến thức tổng hợp của mô hình, chỉ dùng để giải thích/kết nối khi cần; không gắn citation giả.

CÁCH SUY LUẬN CA BỆNH:
- Trước khi kết luận, tự phân rã vấn đề: chỉ định, liều, chức năng thận/gan, tuổi/cân nặng, tương tác thuốc-thuốc, tương tác thuốc-bệnh nhân, tương kỵ pha/truyền, độc tính cộng gộp, monitoring, dữ liệu còn thiếu.
- Phân biệt rõ TƯƠNG KỴ VẬT LÝ/PHA TRUYỀN với CHỐNG CHỈ ĐỊNH hoặc TƯƠNG TÁC LÂM SÀNG. Tương kỵ pha chung không tự động có nghĩa hai thuốc không được dùng trên cùng bệnh nhân.
- Với nhiều thuốc, đánh giá các cặp liên quan và nguy cơ cộng gộp (ví dụ độc thận/độc tai/QT/chảy máu) rồi mới tổng hợp.
- Với định liều, dùng giá trị từ CALCULATION để xác định bệnh nhân nằm ở nhánh/hàng nào của bảng hoặc quy tắc trong HDSD/DUOC_THU/PUBMED. Không suy ra liều chỉ từ eGFR/CrCl nếu evidence không chứa quy tắc liều phù hợp.
- Phương pháp tính được chính nguồn của thuốc dùng trong nghiên cứu/nhãn/bảng liều phải được ưu tiên hơn quy tắc chung. Nếu nguồn không nói rõ, trình bày bất định thay vì tự chọn một công thức với vẻ chắc chắn.
- Nếu AKI/non-steady-state, phù/cổ trướng, sarcopenia, cụt chi, ICU/ARC, CRRT/IHD hoặc kiểu hình khác làm phép ước tính kém tin cậy, phải phản ánh điều đó trong khuyến nghị và confidence; không biến một con số ước tính thành “sự thật”.
- Nếu thiếu dữ liệu, VẪN trả lời phần có thể trả lời và nói rõ điều gì chưa thể chốt; không trả kiểu “hãy xem tài liệu”.
- Nếu options.continuation=true và câu hiện tại chủ yếu là BỔ SUNG DỮ KIỆN (ví dụ chỉ thêm creatinin, chiều cao, xét nghiệm, thuốc mới), KHÔNG chỉ xác nhận “đã nhận”. Hãy dùng hồ sơ ca bệnh đã hợp nhất + HISTORY để ĐÁNH GIÁ LẠI câu hỏi/quyết định lâm sàng trước đó và nêu rõ điều gì thay đổi sau dữ kiện mới.

TRÍCH DẪN VÀ AN TOÀN:
- Mọi item basis="evidence" phải có ít nhất một source_id có thật trong EVIDENCE và nội dung source phải trực tiếp hỗ trợ luận điểm.
- PATIENT_FILE có thể citation cho dữ kiện ca bệnh; CALCULATION có thể citation cho phép tính; khuyến cáo y khoa phải được neo vào HDSD_BV/DUOC_THU/PUBMED/PMC hoặc được ghi rõ basis="ai_knowledge" nếu chỉ là diễn giải tổng hợp.
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
    {"title":"...","items":[{"text":"...","basis":"evidence|ai_knowledge","source_ids":["ID"],"confidence":"high|moderate|low"}]}
  ],
  "alerts": [{"text":"...","source_ids":["ID"]}],
  "evidence_assessment": "đánh giá nguồn/bằng chứng đã dùng",
  "conflicts": [{"text":"...","source_ids":["ID1","ID2"]}],
  "limitations": "giới hạn hiện tại",
  "need_more_data": ["dữ liệu bổ sung có thể làm thay đổi quyết định"]
}`;
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
  const sres = await fetch(searchUrl, { headers: { 'user-agent': 'MedPlusAIPro/0.4' } });
  if (!sres.ok) throw new Error(`NCBI_ESEARCH_HTTP_${sres.status}`);
  const sdata = await sres.json();
  const ids = sdata?.esearchresult?.idlist || [];
  if (!ids.length) return { query, ids: [], xml: '' };

  const fparams = ncbiBaseParams(env);
  fparams.set('db', 'pubmed');
  fparams.set('id', ids.join(','));
  fparams.set('retmode', 'xml');
  const fetchUrl = `https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?${fparams}`;
  const fres = await fetch(fetchUrl, { headers: { 'user-agent': 'MedPlusAIPro/0.4' } });
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
  const ires = await fetch(idUrl, { headers: { 'user-agent': 'MedPlusAIPro/0.4' } });
  if (!ires.ok) throw new Error(`PMC_IDCONV_HTTP_${ires.status}`);
  const idData = await ires.json();
  const mapping = (idData.records || []).filter(r => r.pmcid && r.pmid).map(r => ({ pmid: String(r.pmid), pmcid: String(r.pmcid), doi: r.doi || '' }));
  if (!mapping.length) return { mapping: [], xml: '' };

  const fp = ncbiBaseParams(env);
  fp.set('db', 'pmc');
  fp.set('id', mapping.map(x => x.pmcid).join(','));
  fp.set('retmode', 'xml');
  const url = `https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?${fp}`;
  const res = await fetch(url, { headers: { 'user-agent': 'MedPlusAIPro/0.4' } });
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
        return json({ ok: true, service: 'MedPlus AI Pro', platform: 'Cloudflare Workers', version: '0.4.0', gemini_models: getModels(env), gemini_key_count: getKeys(env).length, ncbi_key: !!env.NCBI_API_KEY, case_state_merge: true }, 200, headers);
      }
      if (request.method !== 'POST') return json({ error: 'METHOD_NOT_ALLOWED' }, 405, headers);
      const uploadRoute = url.pathname === '/api/ai/transcribe' || url.pathname === '/api/ai/extract-file';
      const body = await readJsonBody(request, uploadRoute ? 18_000_000 : MAX_REQUEST_BYTES);

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
