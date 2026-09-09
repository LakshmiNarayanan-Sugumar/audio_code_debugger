import ast
import os

def split_large_chunk(code, max_lines=25):
    lines = code.splitlines()
    if len(lines) <= max_lines:
        return [code]

    blocks, current = [], []
    for line in lines:
        current.append(line)
        if line.strip() == "":
            blocks.append(current)
            current = []
    if current:
        blocks.append(current)

    sub_chunks, buffer, buffer_len = [], [], 0
    for block in blocks:
        if buffer_len + len(block) > max_lines and buffer:
            sub_chunks.append("\n".join(buffer))
            buffer, buffer_len = [], 0
        buffer.extend(block)
        buffer_len += len(block)
    if buffer:
        sub_chunks.append("\n".join(buffer))
    return sub_chunks


def chunk_file(filepath):
    with open(filepath, "r") as f:
        source = f.read()
    tree = ast.parse(source)
    chunks = []
    consumed_lines = set()

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            code = ast.get_source_segment(source, node)
            chunks.append({
                "file": os.path.basename(filepath),
                "name": node.name,
                "type": type(node).__name__,
                "code": code,
            })
            for line in range(node.lineno, node.end_lineno + 1):
                consumed_lines.add(line)

    lines = source.splitlines()
    remaining_lines = [
        line for i, line in enumerate(lines, start=1)
        if i not in consumed_lines
    ]
    remaining_code = "\n".join(remaining_lines)

    if remaining_code.strip():
        sub_chunks = split_large_chunk(remaining_code)
        for i, sub in enumerate(sub_chunks, start=1):
            name = "module_level" if len(sub_chunks) == 1 else f"module_level_{i}"
            chunks.append({
                "file": os.path.basename(filepath),
                "name": name,
                "type": "Module",
                "code": sub,
            })

    return chunks


def chunk_corpus(corpus_dir):
    all_chunks = []
    for fname in sorted(os.listdir(corpus_dir)):
        if fname.endswith(".py"):
            filepath = os.path.join(corpus_dir, fname)
            all_chunks.extend(chunk_file(filepath))
    return all_chunks