async function uploadPDFs() {
    const input = document.getElementById('pdfUpload');
    const formData = new FormData();
  
    for (const file of input.files) {
      formData.append('files', file);
    }
  
    const res = await fetch("/upload", {
      method: "POST",
      body: formData
    });
  
    const data = await res.json();
    document.getElementById('uploadStatus').innerText = data.status;
  }
  
  async function sendQuery() {
    const query = document.getElementById('queryInput').value;
    const formData = new FormData();
    formData.append('query', query);
  
    const res = await fetch("/query", {
      method: "POST",
      body: formData
    });
  
    const data = await res.json();
    const box = document.getElementById('answerBox');
    box.innerHTML = `<h3>🧠 LLaMA 4 Answer:</h3><p>${data.answer}</p>`;
    box.classList.add("slide-in-bottom");
  }
  