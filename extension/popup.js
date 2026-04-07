const backendInput = document.getElementById("backendUrl");
const processBtn = document.getElementById("processBtn");
const askBtn = document.getElementById("askBtn");
const saveBtn = document.getElementById("saveBtn");
const questionInput = document.getElementById("questionInput");
const processStatus = document.getElementById("processStatus");
const answerOutput = document.getElementById("answerOutput");
const sourcesOutput = document.getElementById("sourcesOutput");
const notesOutput = document.getElementById("notesOutput");
const notesFormat = document.getElementById("notesFormat");

let cachedPageId = null;
let cachedSignature = null;

const setStatus = (message) => {
  processStatus.textContent = message;
};

const getActiveTab = async () => {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (!tab?.id) {
    throw new Error("No active tab available");
  }
  return tab;
};

function collectPageData() {
  return {
    url: window.location.href,
    title: document.title || "",
    text: document.body?.innerText || "",
    html: document.body?.innerHTML || ""
  };
}

const extractPage = async () => {
  const tab = await getActiveTab();
  try {
    const response = await chrome.tabs.sendMessage(tab.id, { type: "EXTRACT_PAGE" });
    if (response?.ok) {
      return response.payload;
    }
  } catch (_error) {
    // Fall through to direct injection when the content script is not attached.
  }

  const [result] = await chrome.scripting.executeScript({
    target: { tabId: tab.id },
    func: collectPageData
  });

  if (!result?.result?.text) {
    throw new Error("Could not read page content from this tab");
  }

  return result.result;
};

const signatureFor = (page) => `${page.url}::${page.title}::${page.text.slice(0, 2000)}`;

const postJson = async (path, body) => {
  const response = await fetch(`${backendInput.value}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body)
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(errorText || `Request failed with ${response.status}`);
  }

  return response.json();
};

const ensureProcessed = async () => {
  const page = await extractPage();
  const signature = signatureFor(page);

  if (cachedPageId && cachedSignature === signature) {
    return { page, pageId: cachedPageId, reused: true };
  }

  setStatus("Processing page for retrieval...");
  const result = await postJson("/process", page);
  cachedPageId = result.page_id;
  cachedSignature = signature;
  setStatus(`Indexed ${result.chunk_count} chunks across ${result.sections.length} sections.`);
  return { page, pageId: cachedPageId, reused: false };
};

processBtn.addEventListener("click", async () => {
  try {
    await ensureProcessed();
  } catch (error) {
    setStatus(error instanceof Error ? error.message : "Processing failed");
  }
});

askBtn.addEventListener("click", async () => {
  const question = questionInput.value.trim();
  if (!question) {
    answerOutput.textContent = "Enter a question first.";
    return;
  }

  try {
    const { pageId } = await ensureProcessed();
    answerOutput.textContent = "Thinking...";
    sourcesOutput.innerHTML = "";

    const response = await postJson("/ask", { page_id: pageId, question, top_k: 4 });
    answerOutput.textContent = response.answer;

    if (response.sources.length === 0) {
      const li = document.createElement("li");
      li.textContent = "No supporting chunk returned.";
      sourcesOutput.appendChild(li);
      return;
    }

    response.sources.forEach((source) => {
      const li = document.createElement("li");
      li.textContent = source;
      sourcesOutput.appendChild(li);
    });
  } catch (error) {
    answerOutput.textContent = error instanceof Error ? error.message : "Question failed";
  }
});

saveBtn.addEventListener("click", async () => {
  try {
    const page = await extractPage();
    notesOutput.textContent = "Generating notes...";
    const response = await postJson("/save", { ...page, format: notesFormat.value });
    notesOutput.textContent = response.output;
  } catch (error) {
    notesOutput.textContent = error instanceof Error ? error.message : "Notes failed";
  }
});
