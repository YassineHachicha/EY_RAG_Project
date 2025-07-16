import pdfplumber

pdf_path = "doc3.pdf"  # nom du fichier PDF
txt_path = "doc3.txt"  # nom du fichier texte de sortie

with pdfplumber.open(pdf_path) as pdf:
    all_text = ""
    for i, page in enumerate(pdf.pages):
        text = page.extract_text()
        if text:
            all_text += f"\n\n--- Page {i+1} ---\n\n{text}"

# Écrire dans un fichier .txt
with open(txt_path, "w", encoding="utf-8") as f:
    f.write(all_text)

print("✅ Extraction terminée :", txt_path)
