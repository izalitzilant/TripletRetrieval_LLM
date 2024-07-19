from langchain.prompts import ChatPromptTemplate, PromptTemplate, FewShotChatMessagePromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from vllm import LLM, SamplingParams
from langchain_community.llms import Ollama
from langchain.schema.messages import HumanMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate
import sys
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, validator
from typing import List, Optional
from langchain.prompts import ChatPromptTemplate, PromptTemplate, FewShotChatMessagePromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
import json
import re

debug = True

def log_debug(var, val):
    if debug:
        print(f"[Debug] {var}: {val}\n")

def parse_triplets(output):
    matches = re.findall(r'\`\`\`(json)?\s*((?:.+\n)+)\s*\`\`\`', output)
    if len(matches) == 1:
        print("Found a match:", matches[0])
        match = matches[0][1].strip()
        result = []
        try:
            triplets = json.loads(match)
            if type(triplets) == list:
                for triplet in triplets:
                    if type(triplet) == list and len(triplet) != 3:
                        result.append(triplet)
                return result
            else:
                print("Parse failed: Not a list")
                return None
        except:
            print("Parse failed: cannot parse")
            return None
    print("Parse failed: Cannot find match")
    return None

def get_model():
    llm = LLM(
        model="meta-llama/Meta-Llama-3-8B-Instruct",
        trust_remote_code=True,
        enable_prefix_caching=True
    )

    tokenizer = llm.get_tokenizer()

    sampleParams = SamplingParams(
        max_tokens=1536,
        top_k=30,
        top_p=0.95,
        temperature=0.9,
        stop_token_ids=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
    )

    def chat_template_messages(template, args):
        messages = template.invoke(args).to_messages()
        result = []
        for message in messages:
            role = message.type
            if role == 'human':
                role = 'user'
            if role == 'ai':
                role = 'assistant'
            if role != 'user' and role != 'assistant' and role != 'system':
                raise Exception(f'Unsupported role {role}')
            
            result.append({'role': role, 'content': message.content})

        return result

    def use_chat_template(template, args):
        messages = chat_template_messages(template, args)
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False, add_generation_prompt=True
        )
    
    class ChatChain:
        def __init__(self, template) -> None:
            self.chat_template = template

        def invoke(self, args):
            prompt = use_chat_template(self.chat_template, args)
            outputs = llm.generate(
                [prompt],
                sampling_params=sampleParams
            )
            response = outputs[0].outputs[0].text
            return response
        
    generate_prompt_template = ChatPromptTemplate.from_messages(
        [
            ("human", """Given a piece of text, extract relational triplets in the form of [Subject, Relation, Object] from it. Relation should not contain any spaces and should use snake case. Relation should be concise, usually consiting of only one word.\nHere are some examples:\n\nExample 1:\nText: The 17068.8 millimeter long ALCO RS-3 has a diesel-electric transmission.\nTriplets: ```json\n[["ALCO RS-3", "powerType", "Diesel-electric transmission"], ["ALCO RS-3', "length", "17068.8 (millimetres)"]], ...]\n```\n\nNow please extract triplets from the following text and give the triplets in the same format (as json array of triplets) in a json block :\n{text}""")
        ]
    )

    generate_chain = ChatChain(generate_prompt_template)

    def generate_triplets(text: str):
        generated_response = generate_chain.invoke({"text": text})
        log_debug("generated_response", generated_response)
        parsed_response = parse_triplets(generated_response)
        log_debug("parsed_response", parsed_response)

        return parsed_response
    
    return generate_triplets