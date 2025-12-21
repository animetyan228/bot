#import fitz  # PyMuPDF

#ф-ция чтения пдф
#def read_pdf(pdf_path: str) -> str:
 #   doc = fitz.Document(pdf_path)  # открываем
  #  text = ""  # куда мы будем помещать текст
   # for page in doc:  # цикл по страницам пдф
    #    text += page.get_text() + "\n"  # достаем текст с одной страницы
    #return text  # возращаем прочитаный пдф в текст
import fitz

def read_pdf(pdf_path: str) -> str:
    doc = fitz.Document(pdf_path) # открываем pdf правильно

    all_text = []
    for i in range(doc.page_count):
        page = doc.load_page(i)

        # пробуем обычный текст
        t = page.get_text("text").strip()

        # если текста мало, пробуем "blocks" (иногда даёт больше)
        if len(t) < 50:
            blocks = page.get_text("blocks")
            t = "\n".join([b[4] for b in blocks if b[4].strip()]).strip()

        all_text.append(t)

    doc.close()
    return "\n\n".join([x for x in all_text if x])
