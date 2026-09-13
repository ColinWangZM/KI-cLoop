'''
This script takes the MCQ style questions from the csv file and save the result as another csv file. 
This script makes use of Llama model.
Before running this script, make sure to configure the filepaths in config.yaml file.
'''
import os
import re
from difflib import SequenceMatcher
from langchain import PromptTemplate, LLMChain
from kg_rag.utility import *
from kg_rag.rag_based_generation.Llama.application_synthesis_qa import answer_application_synthesis_question
from kg_rag.rag_based_generation.Llama.kinetics_qa import answer_kinetics_question
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel, PeftConfig

# 设置模型和 LoRA 路径
BASE_MODEL_PATH = BASE_QWEN_MODEL_PATH
LORA_PATH = OR_LORA_PATH

QUESTION_PATH = config_data["MCQ_PATH_or"]
SYSTEM_PROMPT = system_prompts["MCQ_QUESTION_or"]#system_prompts.yaml文件中MCQ_QUESTION的值
CONTEXT_VOLUME = int(config_data["CONTEXT_VOLUME"])#上下文体积
QUESTION_VS_CONTEXT_SIMILARITY_PERCENTILE_THRESHOLD = float(config_data["QUESTION_VS_CONTEXT_SIMILARITY_PERCENTILE_THRESHOLD"])#衡量问题与上下文之间相似度的百分位数阈值
QUESTION_VS_CONTEXT_MINIMUM_SIMILARITY = float(config_data["QUESTION_VS_CONTEXT_MINIMUM_SIMILARITY"])#衡量问题与上下文之间相似度的最小相似度
VECTOR_DB_PATH = config_data["VECTOR_DB_PATH_OR"]#向量数据库路径
NODE_CONTEXT_PATH = os.environ.get(
    "KGRAG_OR_CONTEXT",
    config_data["NODE_CONTEXT_PATH_OR"],
)
SENTENCE_EMBEDDING_MODEL_FOR_NODE_RETRIEVAL = config_data["SENTENCE_EMBEDDING_MODEL_FOR_NODE_RETRIEVAL"]
SENTENCE_EMBEDDING_MODEL_FOR_CONTEXT_RETRIEVAL = config_data["SENTENCE_EMBEDDING_MODEL_FOR_CONTEXT_RETRIEVAL"]
SAVE_PATH = config_data["SAVE_RESULTS_PATH"]
MODEL_NAME = config_data["LLAMA_MODEL_NAME"]
BRANCH_NAME = config_data["LLAMA_MODEL_BRANCH"]
CACHE_DIR = config_data["LLM_CACHE_DIR"]

save_name = "_".join(MODEL_NAME.split("/")[-1].split("-"))+"_or_mcq_done0509_promptfix.csv"
MAX_QUESTIONS = int(os.environ.get("KGRAG_MAX_QUESTIONS", "500"))


INSTRUCTION = "Context:\n\n{context} \n\nQuestion: {question}"

vectorstore = None
embedding_function_for_context_retrieval = None
node_context_df = pd.read_csv(NODE_CONTEXT_PATH)
edge_evidence = False
GENERIC_REACTION_TERMS = {
    "",
    "acid",
    "base",
    "ice",
    "ice water",
    "water",
    "product",
    "solution",
    "aqueous solution",
}


def ensure_vector_retrieval_loaded():
    global vectorstore
    global embedding_function_for_context_retrieval
    if vectorstore is None:
        vectorstore = load_chroma(
            VECTOR_DB_PATH,
            SENTENCE_EMBEDDING_MODEL_FOR_NODE_RETRIEVAL,
        )
    if embedding_function_for_context_retrieval is None:
        embedding_function_for_context_retrieval = load_sentence_transformer(
            SENTENCE_EMBEDDING_MODEL_FOR_CONTEXT_RETRIEVAL
        )

def load_model():
    # 加载基础模型
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH,
                                            revision=BRANCH_NAME,
                                            cache_dir=CACHE_DIR,
                                            trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        **model_load_kwargs(BRANCH_NAME, CACHE_DIR),
    )
    
    # 加载 LoRA 权重
    model = PeftModel.from_pretrained(model, LORA_PATH)
    
    # 创建 pipeline
    pipe = pipeline("text-generation",
                model=model,
                tokenizer=tokenizer,
                **pipeline_kwargs(max_new_tokens=768))
    
    return HuggingFacePipeline(pipeline=pipe,
                             model_kwargs={"temperature":0, "top_p":1})

def extract_quoted_target(question):
    matches = re.findall(r'"([^"]+)"', question)
    return matches[-1].strip() if matches else ""


def normalize_name(name):
    return re.sub(r"[^a-z0-9]+", "", str(name).casefold())


def canonical_node_name(name):
    normalized_target = normalize_name(name)
    if not normalized_target:
        return ""

    node_names = node_context_df["node_name"].astype(str)
    exact_match = node_context_df[node_names.str.casefold() == str(name).casefold()]
    if not exact_match.empty:
        return exact_match.iloc[0]["node_name"]

    normalized_names = node_names.map(normalize_name)
    normalized_match = node_context_df[normalized_names == normalized_target]
    if not normalized_match.empty:
        return normalized_match.iloc[0]["node_name"]

    fuzzy_scores = normalized_names.map(
        lambda node_name: SequenceMatcher(None, normalized_target, node_name).ratio()
    )
    best_index = fuzzy_scores.idxmax()
    if fuzzy_scores.loc[best_index] >= 0.92:
        return node_context_df.loc[best_index, "node_name"]
    return ""


def get_context_for_or_question(question):
    target = extract_quoted_target(question)
    if target:
        normalized_target = normalize_name(target)
        exact_match = node_context_df[
            node_context_df["node_name"].astype(str).str.casefold() == target.casefold()
        ]
        if not exact_match.empty:
            return exact_match.iloc[0]["node_context"], exact_match.iloc[0]["node_name"], 0.0, "exact_node_name"

        node_names = node_context_df["node_name"].astype(str)
        normalized_names = node_names.map(normalize_name)
        normalized_match = node_context_df[normalized_names == normalized_target]
        if not normalized_match.empty:
            return (
                normalized_match.iloc[0]["node_context"],
                normalized_match.iloc[0]["node_name"],
                0.0,
                "normalized_node_name",
            )

        fuzzy_scores = normalized_names.map(
            lambda name: SequenceMatcher(None, normalized_target, name).ratio()
        )
        best_index = fuzzy_scores.idxmax()
        if fuzzy_scores.loc[best_index] >= 0.92:
            return (
                node_context_df.loc[best_index, "node_context"],
                node_context_df.loc[best_index, "node_name"],
                float(fuzzy_scores.loc[best_index]),
                "fuzzy_node_name",
            )

    ensure_vector_retrieval_loaded()
    node_hits = vectorstore.similarity_search_with_score(question, k=3)
    for document, score in node_hits:
        node_name = document.page_content
        node_match = node_context_df[node_context_df.node_name == node_name]
        if not node_match.empty:
            return node_match.iloc[0]["node_context"], node_name, float(score), "vector_top3"

    context = retrieve_context(
        question,
        vectorstore,
        embedding_function_for_context_retrieval,
        node_context_df,
        CONTEXT_VOLUME,
        QUESTION_VS_CONTEXT_SIMILARITY_PERCENTILE_THRESHOLD,
        QUESTION_VS_CONTEXT_MINIMUM_SIMILARITY,
        edge_evidence,
    )
    return context, "", np.nan, "legacy_retrieve_context"


def build_reaction_graph():
    adjacency = {}
    direct_edges = {}
    for _, row in node_context_df.iterrows():
        product = str(row["node_name"])
        product_context = str(row["node_context"])
        for reactant in context_values(product_context, "反应物名称:"):
            if reactant.casefold() in GENERIC_REACTION_TERMS:
                continue
            edge = {
                "source": reactant,
                "target": product,
                "reactants": context_values(product_context, "反应物名称:"),
                "operation_steps": operation_steps(product_context),
                "context": product_context,
            }
            source_key = normalize_name(reactant)
            target_key = normalize_name(product)
            adjacency.setdefault(source_key, []).append(edge)
            direct_edges.setdefault((source_key, target_key), []).append(edge)
    return adjacency, direct_edges


REACTION_ADJACENCY = {}
DIRECT_REACTION_EDGES = {}


def direct_reaction_edge(source, target):
    edges = DIRECT_REACTION_EDGES.get((normalize_name(source), normalize_name(target)), [])
    return edges[0] if edges else None


def find_reaction_paths(source, target, max_depth=2, max_paths=5, min_depth=1):
    source_key = normalize_name(source)
    target_key = normalize_name(target)
    if not source_key or not target_key:
        return []

    paths = []
    queue = [(source_key, [], {source_key})]
    while queue:
        current_key, path, seen = queue.pop(0)
        if len(path) >= max_depth:
            continue
        for edge in REACTION_ADJACENCY.get(current_key, []):
            next_key = normalize_name(edge["target"])
            new_path = path + [edge]
            if next_key == target_key and len(new_path) >= min_depth:
                paths.append(new_path)
                if len(paths) >= max_paths:
                    return paths
            if next_key not in seen:
                queue.append((next_key, new_path, seen | {next_key}))
    return paths


def extract_quoted_entities(question):
    return [item.strip() for item in re.findall(r'"([^"]+)"', str(question))]


def parse_arrow_path(question):
    if "->" not in str(question):
        return []
    parts = [part.strip(" .?\"'") for part in str(question).split("->")]
    return [part for part in parts if part]


def parse_multihop_query(question):
    question_text = str(question)
    question_lower = question_text.lower()
    quoted = extract_quoted_entities(question_text)

    arrow_path = parse_arrow_path(question_text)
    if len(arrow_path) >= 2:
        return {"kind": "explicit_path", "nodes": arrow_path}

    from_to = re.search(r'from\s+"([^"]+)".*to\s+"([^"]+)"', question_text, flags=re.I)
    if from_to:
        min_depth = 2 if "multi-step" in question_lower or "multistep" in question_lower else 1
        return {
            "kind": "source_target",
            "source": from_to.group(1),
            "target": from_to.group(2),
            "min_depth": min_depth,
        }

    if len(quoted) >= 3 and ("generated by" in question_lower or "can be generated by" in question_lower):
        return {"kind": "explicit_path", "nodes": list(reversed(quoted[:3]))}

    if len(quoted) >= 2 and (
        "generated from" in question_lower
        or "synthesized from" in question_lower
        or "produced from" in question_lower
        or "precursor" in question_lower
        or "multi-step" in question_lower
        or "multistep" in question_lower
    ):
        min_depth = 2 if "multi-step" in question_lower or "multistep" in question_lower else 1
        return {"kind": "source_target", "source": quoted[1], "target": quoted[0], "min_depth": min_depth}

    return {}


def format_reaction_step(edge, step_index):
    answer = [f"Step {step_index}: {edge['source']} -> {edge['target']}"]
    if edge["reactants"]:
        answer.append("Reactants: " + ", ".join(edge["reactants"]))
    if edge["operation_steps"]:
        answer.append("Operation steps: " + " ".join(edge["operation_steps"]))
    return "\n".join(answer)


def format_reaction_path_answer(paths):
    if not paths:
        return ""
    lines = [f"Yes. Found {len(paths)} reaction path(s) in the current reaction graph."]
    for path_index, path in enumerate(paths, start=1):
        lines.append(f"\nPath {path_index}:")
        for step_index, edge in enumerate(path, start=1):
            lines.append(format_reaction_step(edge, step_index))
    return "\n".join(lines)


def multihop_reaction_answer(question):
    parsed = parse_multihop_query(question)
    if not parsed:
        return "", "", "", ""

    if parsed["kind"] == "explicit_path":
        nodes = parsed["nodes"]
        path = []
        for source, target in zip(nodes[:-1], nodes[1:]):
            canonical_source = canonical_node_name(source) or source
            canonical_target = canonical_node_name(target) or target
            edge = direct_reaction_edge(canonical_source, canonical_target)
            if not edge:
                return (
                    f"No direct edge found for {canonical_source} -> {canonical_target} in the current reaction graph.",
                    canonical_target,
                    "reaction_path_explicit_miss",
                    "",
                )
            path.append(edge)
        return format_reaction_path_answer([path]), nodes[-1], "reaction_path_explicit", "\n\n".join(
            edge["context"] for edge in path
        )

    if parsed["kind"] == "source_target":
        source = canonical_node_name(parsed["source"]) or parsed["source"]
        target = canonical_node_name(parsed["target"]) or parsed["target"]
        paths = find_reaction_paths(
            source,
            target,
            max_depth=3,
            max_paths=5,
            min_depth=parsed.get("min_depth", 1),
        )
        if not paths:
            return (
                f"No reaction path from {source} to {target} was found within 3 steps in the current reaction graph.",
                target,
                "reaction_path_search_miss",
                "",
            )
        return format_reaction_path_answer(paths), target, "reaction_path_search", "\n\n".join(
            edge["context"] for path in paths for edge in path
        )

    return "", "", "", ""


def clean_output(output):
    output = str(output).strip()
    if "Answer:" in output:
        output = output.split("Answer:")[-1].strip()
    stop_markers = ["\nQuestion:", "\nContext:", "\nNow,"]
    for marker in stop_markers:
        if marker in output:
            output = output.split(marker)[0].strip()
    return output


def unique_values(values):
    seen = set()
    unique = []
    for value in values:
        value = value.strip().strip(".")
        if value and value.casefold() not in seen:
            unique.append(value)
            seen.add(value.casefold())
    return unique


def context_values(context, field_name):
    values = []
    for line in str(context).splitlines():
        line = line.strip()
        if line.startswith(field_name):
            values.append(line.split(":", 1)[1].strip())
    return unique_values(values)


def operation_steps(context):
    steps = []
    for line in str(context).splitlines():
        line = line.strip()
        if line.startswith("操作步骤:"):
            steps.append(line.split(":", 1)[1].strip())
    return unique_values(steps)


REACTION_ADJACENCY, DIRECT_REACTION_EDGES = build_reaction_graph()


def rule_based_or_answer(question, context):
    question_lower = str(question).lower()
    reactants = context_values(context, "反应物名称:")
    steps = operation_steps(context)
    if (
        "reactants for generating" in question_lower
        or "reactants for" in question_lower
        or "what are the reactants" in question_lower
        or "反应物" in str(question)
    ):
        return ", ".join(reactants) if reactants else "Not found in retrieved context."
    if (
        "operation steps" in question_lower
        or "operation" in question_lower
        or "steps" in question_lower
        or "操作" in str(question)
        or "步骤" in str(question)
    ):
        return " ".join(steps) if steps else "Not found in retrieved context."
    if "synthesis route" in question_lower or "how to generate" in question_lower:
        parts = []
        if reactants:
            parts.append("Reactants: " + ", ".join(reactants))
        if steps:
            parts.append("Operation steps: " + " ".join(steps))
        return "; ".join(parts) if parts else "Not found in retrieved context."
    return ""


#2questions
def main():
    start_time = time.time()
    

    template = """
You are an organic reaction information extraction assistant.
Use ONLY the retrieved context. Do not use outside knowledge. Do not copy the example from another reaction.

Task rules:
1. If the question asks "What are the reactants for generating ...", answer ONLY with the values after "反应物名称:" from the context.
   - Return a comma-separated list of reactant names.
   - Do NOT include product names, solvents, quantities, SMILES, InChI, operation labels, workup materials, or yields.
2. If the question asks "What is the synthesis route of ..." or "How to generate ...", extract the route from the same product context.
   - Include the reactant names from "反应物名称:".
   - Include the operation text after "操作步骤:".
   - Preserve important conditions such as temperature, time, amount, solvent, washing, drying, purification, and yield if they appear in the context.
   - Do NOT invent missing steps.
3. If the answer is not present in the context, output exactly: Not found in retrieved context.
4. Return only the final answer, with no explanation and no JSON wrapper unless the question explicitly requests JSON.

Context:
{context}

Question: {question}

Answer:"""
    
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])#"important_instructions"
    
    question_df = pd.read_csv(QUESTION_PATH)  
    
    # 检查文件是否存在并确定起始位置
    save_file_path = os.path.join(SAVE_PATH, save_name)
    if os.path.exists(save_file_path):
        existing_df = pd.read_csv(save_file_path)
        question_count = len(existing_df)
        print(f"发现已存在的文件，从第 {question_count + 1} 个问题继续...")
    else:
        question_count = 0
        # 创建新文件并写入表头
        pd.DataFrame(columns=[
            "question",
            "correct_answer",
            "llm_answer",
            "retrieved_node",
            "retrieval_score",
            "retrieval_method",
            "context",
        ]).to_csv(
            save_file_path, index=False, mode='w'
        )
    
    llm_chain = None
    
    target_count = min(MAX_QUESTIONS, len(question_df))
    for index, row in question_df.iloc[question_count:].iterrows():
        if question_count >= target_count:
            break
        print("##########################")
        print(f"正在处理第 {question_count + 1} 个问题")
        print("使用Chemllm进行KG-RAG推理")
        
        question = row["text"]
        # print("question:",question)
        



        # # 在检索之前添加调试代码
        # print("Available nodes:", node_context_df['node_name'].unique())
        # print("Searching for node:", "TERT-BUTYL HYDROPEROXIDE")




        retrieved_node = ""
        retrieval_method = ""
        context = ""
        output = answer_application_synthesis_question(question)
        if output:
            retrieved_node = "material_application_index"
            retrieval_method = "application_synthesis_rule_qa"
            context = output
        else:
            output = answer_kinetics_question(question)
        if output:
            if not retrieved_node:
                retrieved_node = "kinetics_route_knowledge"
                retrieval_method = "kinetics_rule_qa"
                context = output
        else:
            output, retrieved_node, retrieval_method, context = multihop_reaction_answer(question)
        retrieval_score = np.nan

        if not output:
            context, retrieved_node, retrieval_score, retrieval_method = get_context_for_or_question(question)
            rule_answer = rule_based_or_answer(question, context)
            if rule_answer:
                output = rule_answer
            else:
                if llm_chain is None:
                    llm = load_model()
                    llm_chain = LLMChain(prompt=prompt, llm=llm)
                output = llm_chain.run(
                    context=context,
                    question=question
                )
                output = clean_output(output)
        
        # 立即将当前结果追加到文件
        result_df = pd.DataFrame({
            "question": [row["text"]],
            "correct_answer": [row["correct_ans"]],
            "llm_answer": [output],
            "retrieved_node": [retrieved_node],
            "retrieval_score": [retrieval_score],
            "retrieval_method": [retrieval_method],
            "context": [context],
        })

        #print("context:",context)

        # 追加结果到CSV文件
        result_df.to_csv(save_file_path, mode='a', header=False, index=False)
        print("问题内容：", row["text"])
        print("检索节点：", retrieved_node)
        print("检索方式：", retrieval_method)
        print("正确答案：", row["correct_ans"])
        print("推理答案：", output)
        print("##########################")
        
        question_count += 1

    print("Completed in {} min".format((time.time()-start_time)/60))


if __name__ == "__main__":
    main()
    
    
