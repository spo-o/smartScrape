chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (message?.type !== "EXTRACT_PAGE") {
    return false;
  }

  try {
    sendResponse({
      ok: true,
      payload: {
        url: window.location.href,
        title: document.title || "",
        text: document.body?.innerText || "",
        html: document.body?.innerHTML || ""
      }
    });
  } catch (error) {
    sendResponse({
      ok: false,
      error: error instanceof Error ? error.message : "Failed to extract page"
    });
  }

  return false;
});
