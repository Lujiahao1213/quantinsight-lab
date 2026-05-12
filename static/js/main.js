console.log("QuantInsight Lab loaded.");

const fileInput = document.getElementById("fileInput");
const uploadDropzone = document.getElementById("uploadDropzone");
const selectedFileName = document.getElementById("selectedFileName");

if (fileInput && uploadDropzone && selectedFileName) {
  const updateSelectedFileName = () => {
    const file = fileInput.files && fileInput.files[0];
    selectedFileName.textContent = file ? `Selected: ${file.name}` : "No file selected";
  };

  uploadDropzone.addEventListener("click", (event) => {
    if (event.target !== fileInput) {
      fileInput.click();
    }
  });

  uploadDropzone.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      fileInput.click();
    }
  });

  fileInput.addEventListener("change", updateSelectedFileName);

  ["dragenter", "dragover"].forEach((eventName) => {
    uploadDropzone.addEventListener(eventName, (event) => {
      event.preventDefault();
      uploadDropzone.classList.add("dragover");
    });
  });

  ["dragleave", "dragend"].forEach((eventName) => {
    uploadDropzone.addEventListener(eventName, () => {
      uploadDropzone.classList.remove("dragover");
    });
  });

  uploadDropzone.addEventListener("drop", (event) => {
    event.preventDefault();
    uploadDropzone.classList.remove("dragover");

    const droppedFiles = event.dataTransfer?.files;
    if (!droppedFiles || droppedFiles.length === 0) {
      return;
    }

    const dataTransfer = new DataTransfer();
    dataTransfer.items.add(droppedFiles[0]);
    fileInput.files = dataTransfer.files;
    updateSelectedFileName();
  });
}

const strategySelect = document.getElementById("strategy_name");
const strategyParamCards = document.querySelectorAll("[data-strategy-params]");

if (strategySelect && strategyParamCards.length > 0) {
  const updateStrategyParamCards = () => {
    strategyParamCards.forEach((card) => {
      const isVisible = card.dataset.strategyParams === strategySelect.value;
      card.style.display = isVisible ? "block" : "none";
      card.querySelectorAll("input, select").forEach((field) => {
        field.disabled = !isVisible;
      });
    });
  };

  strategySelect.addEventListener("change", updateStrategyParamCards);
  updateStrategyParamCards();
}

const resizePlotlyCharts = () => {
  window.dispatchEvent(new Event("resize"));

  if (!window.Plotly) {
    return;
  }

  document.querySelectorAll(".js-plotly-plot").forEach((plot) => {
    try {
      window.Plotly.Plots.resize(plot);
    } catch (error) {
      console.warn("Plotly resize failed:", error);
    }
  });
};

window.addEventListener("load", () => {
  setTimeout(resizePlotlyCharts, 300);
  setTimeout(resizePlotlyCharts, 1000);
});
