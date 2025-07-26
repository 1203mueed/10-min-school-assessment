from bangla_pdf_ocr import process_pdf

input_pdf = "HSC26_Bangla_1st_paper.pdf"
output_file = "data/output.txt"
language = "ben"  # 'ben' is the ISO code for Bengali

extracted_text = process_pdf(input_pdf, output_file, language)
print(f"Text extracted and saved to: {output_file}\nSample: {extracted_text[:500]}")