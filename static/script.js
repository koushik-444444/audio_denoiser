document.addEventListener("DOMContentLoaded", () => {
    const dropZone = document.getElementById("drop-zone");
    const inputFile = document.getElementById("audio_file");

    dropZone.addEventListener("click", () => {
        inputFile.click();
    });

    inputFile.addEventListener("change", () => {
        dropZone.classList.remove("highlight");
    });

    dropZone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropZone.classList.add("highlight");
    });

    dropZone.addEventListener("dragleave", () => {
        dropZone.classList.remove("highlight");
    });

    dropZone.addEventListener("drop", (e) => {
        e.preventDefault();
        inputFile.files = e.dataTransfer.files;
        dropZone.classList.remove("highlight");
    });
});
