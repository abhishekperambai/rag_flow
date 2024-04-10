import pandas as pd
import json, re
import pathlib
from unstructured.partition.pdf import partition_pdf
from unstructured.staging.base import convert_to_dict
import fitz, os, cv2
import pytesseract
from PIL import Image

class PostProcessor:
    def __init__(self, df):
        self. df = df
    
    def get_chunk(self, x):
        """
        Gets chunk and content from the text provided
        """
        val = x.split("\n")[0].strip()
        if val.isdigit():
            
            return int(val)
        else:
            return None

    def remove_chunk_num(self, x):
        """
        Removes the chunk number as it is separated already
        """
        if x.split("\n")[0].strip().isdigit():
            return "\n".join(x.split("\n")[1:])
        else:
            return x
        
    def group_chunk_contents(self, df):
        """
        Groups the chunk content together as it can have repeated rows in the dataframe
        """
        final = df.groupby("chunk", as_index=False)["content"].apply(lambda x: "\n".join(x))
        final["content"] = final["content"].apply(lambda x: self.remove_chunk_num(x))
        return final
    
    def preprocess_helper(self, x):
        pronouns = ['I', 'me', 'my', 'mine', 'myself']
        for pronoun in pronouns:
            x = re.sub(fr"\b{pronoun}\b", "Nelson Mandela", x)
            # x = x.replace(pronoun, "Nelson Mandela")
        return x
    
    def preprocess(self):
        """
        Works in conjunction with preprocess_helper func to convert the first person 
        elements to the name of the person specified inside the helper func.  
        """
        self.df["content"] = self.df["content"].apply(lambda x: self.preprocess_helper(x))
        return self.df

    def main(self, chunk_flag = False):
        """
        Triggers appropriate methods based on the chunk flag to post process the text.
        """
        if chunk_flag:
            self.df["chunk"] = self.df["content"].apply(lambda x: self.get_chunk(x))
            self.df.loc[0, "chunk"] = 0
            self.df["chunk"] = self.df["chunk"].ffill()
            return self.group_chunk_contents(self.df)
        else:
            return self.preprocess()
        




class PDFParser:
    def __init__(self, fp, images=False):
        self.fp = fp
        self.images = images
    
    def post_process(self,metadata):
        """
        Returns a dictionary with page number as key and content as values.
        """
        list = []
        page_text_dict = {}
        if metadata:
            for item in metadata:
                page_number = item['page_number']
                text = item['text']
                if page_number in page_text_dict:
                    page_text_dict[page_number] += '\n' + text
                else:
                    page_text_dict[page_number] = text
        else:
            page_text_dict[item['page_number']] = ""
        return page_text_dict
    
    def extract_pdf_with_unstructured(self):
        """
        Extracts text and metadata from PDF using unstructured library.
        Then converts it to a pandas dataframe with page number and content as columns. 
        """
        output = {}
        try:
            elements = partition_pdf(filename=self.fp,
                                     infer_table_structure=False
                                     )
            pdf_content = "\n\n".join([str(el) for el in elements])
            meta = convert_to_dict(elements)
            metadata = [{'type': d['type'],
                        "text": d["text"],
                        'page_number': d['metadata']['page_number'],
                        'link': d['metadata']['links'] if 'links' in d['metadata'] else None
                         } for d in meta
                        ]
            output = self.post_process(metadata)
            df = pd.DataFrame([output]).T.reset_index()
            df.columns=["page_num", "content"]
            return df
        except Exception as e:
            print(f"Error processing {self.fp}: {e}")
            return False
        
    def extract_text_from_image(self, image_path):
        """
        This func uses opencv library to identify countours of an image and follows
        the usual protocols to extract text from image.
        """
        full_text = []
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
        rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 18))
        dilation = cv2.dilate(thresh1, rect_kernel, iterations = 1)
        contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, 
                                                    cv2.CHAIN_APPROX_NONE)
        im2 = img.copy()
        for cnt in contours[::-1]:
            x, y, w, h = cv2.boundingRect(cnt)
            rect = cv2.rectangle(im2, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cropped = im2[y:y + h, x:x + w]
            text = pytesseract.image_to_string(cropped)
            if text:
                full_text.append(text.replace("\n\n", "\n"))
        return "".join(full_text)
    
    def extract_pdf_with_fitz(self):
        """
        This func used fitz library to extract text and identify images in the page.
        Works in conjuntion with extract_text_from_image func to perform OCR.
        """
        pdf_document = fitz.open(self.fp)
        extracted_text = []

        for page_num in range(len(pdf_document)):
            dict_ = {}
            print("PAGE_NUM", page_num)
            page = pdf_document.load_page(page_num)
            page_content = ""
            
            page_text = page.get_text()
            page_content += page_text
            
            for img_index, img in enumerate(page.get_images(full=True)):
                xref = img[0]
                base_image = pdf_document.extract_image(xref)
                image_bytes = base_image['image']
                image_extension = base_image['ext']
                
                image_path = f"temp_image_{page_num}_{img_index}.{image_extension}"
                with open(image_path, 'wb') as f:
                    f.write(image_bytes)
                
                image_text = self.extract_text_from_image(image_path)
                page_content += '\n' + image_text
                os.remove(image_path)
            
        dict_["page_num"] = page_num+1
        dict_["content"] = page_content
        extracted_text.append(dict_)
        pdf_document.close()

        df = pd.DataFrame(extracted_text)
        df.columns=["page_num", "content"]
        return df

        
    def parse(self, chunk_flag=False):
        """
        Based on the images flag, the respective functions are executed to extract the text.
        Post processor class is responsible for post processing the extracted text acc to the requirements
        of the project.
        """
        if not self.images:
            df = self.extract_pdf_with_unstructured()
            df.to_csv("./output/extracted_df.csv", index=False)
            processor = PostProcessor(df)
            if chunk_flag:
                processor.main(chunk_flag=chunk_flag).to_csv("./output/extracted_df_chunks.csv", index=False)
            else:
                processor.main(chunk_flag=chunk_flag).to_csv("./output/extracted_df_processed.csv", index=False)
            return True
        else:
            df = self.extract_pdf_with_fitz()
            df.to_csv("./output/extracted_df_fitz.csv", index=False)
            processor = PostProcessor(df)
            if chunk_flag:
                processor.main(chunk_flag=chunk_flag).to_csv("./output/extracted_df_fitz_chunks.csv", index=False)
            else:
                processor.main(chunk_flag=chunk_flag).to_csv("./output/extracted_df_fitz_processed.csv", index=False)
            return True

if __name__ == "__main__":
    fp = "./dataset/Long-Walk-to-Freedom-Autobiography-of-Nelson-Mandela.pdf"
    parser = PDFParser(fp)
    parser.parse(chunk_flag=False)