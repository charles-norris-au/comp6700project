from transformers import pipeline
import torch
import fitz

def extract_pdf_to_file(pdf_path, out_path="pdf_text.txt"):
    """
    Extracts text from a PDF and writes it directly to a file.
    This avoids storing the entire PDF text in memory.
    """
    doc = fitz.open(pdf_path)
    with open(out_path, "w", encoding="utf-8") as f:
        for page in doc:
            f.write(page.get_text())
            f.write("\n")
    return out_path



def chunk_text_file(input_path, chunk_size=4000):
    """
    Splits a large text file into multiple chunk files.
    Each chunk is written to disk to avoid memory usage.
    """
    chunks = []
    current = []
    current_len = 0
    chunk_index = 0

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if current_len + len(line) > chunk_size:
                out_path = f"chunk_{chunk_index}.txt"
                with open(out_path, "w", encoding="utf-8") as out:
                    out.write("".join(current))
                chunks.append(out_path)
                chunk_index += 1
                current = []
                current_len = 0

            current.append(line)
            current_len += len(line)

    # Write last chunk
    if current:
        out_path = f"chunk_{chunk_index}.txt"
        with open(out_path, "w", encoding="utf-8") as out:
            out.write("".join(current))
        chunks.append(out_path)

    return chunks


def summarize_with_gemma(pipe, text, max_new_tokens=50):
    messages = [
        [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You summarize documents clearly and concisely."}]
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": f"Summarize the following:\n\n{text}"}]
            },
        ]
    ]

    output = pipe(messages, max_new_tokens=max_new_tokens)

    # Robust extraction logic
    result = output[0]

    if isinstance(result, list):
        try:
            return result[0]["content"][0]["text"]
        except Exception:
            pass

    if "generated_text" in result:
        gen = result["generated_text"]

        if isinstance(gen, list) and isinstance(gen[0], dict) and "content" in gen[0]:
            try:
                return gen[0]["content"][0]["text"]
            except Exception:
                pass

        if isinstance(gen, str):
            return gen

    return str(result)

def summarize_chunk_from_file(pipe, chunk_path, max_new_tokens=200):
    with open(chunk_path, "r", encoding="utf-8") as f:
        text = f.read()
    return summarize_with_gemma(pipe, text, max_new_tokens=max_new_tokens)


def summarize_pdf_with_gemma_streaming(pdf_path, chunk_size=4000, max_new_tokens=200):
    # 1. Extract PDF → text file
    text_file = extract_pdf_to_file(pdf_path)

    # 2. Chunk text file → chunk files
    chunk_files = chunk_text_file(text_file, chunk_size=chunk_size)
    print(f"PDF split into {len(chunk_files)} chunks")

    # 3. Load Gemma
    pipe = pipeline(
        "text-generation",
        model="google/gemma-3-1b-it",
        device="cpu",
        dtype=torch.bfloat16
    )

    # 4. Summarize each chunk
    chunk_summaries = []
    for i, chunk_path in enumerate(chunk_files, start=1):
        print(f"Summarizing chunk {i}/{len(chunk_files)}...")
        summary = summarize_chunk_from_file(pipe, chunk_path, max_new_tokens)
        chunk_summaries.append(summary)

    # 5. Summarize the summaries
    combined = "\n\n".join(chunk_summaries)
    final_summary = summarize_with_gemma(pipe, combined, max_new_tokens=max_new_tokens)

    return final_summary


summary = summarize_pdf_with_gemma_streaming("cis-r1.pdf")
print(summary)