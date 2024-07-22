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

debug = False

def log_debug(var, val):
    if debug:
        print(f"[Debug] {var}: {val}\n")

def parse_json(output):
    output.split("\n")
    result = []
    json_block = False
    for line in output.split("\n"):
        if line.startswith("```") and not json_block:
            json_block = True
            continue
        if (line.startswith("```") or line.endswith("```")) and json_block:
            json_block = False
            continue
        if json_block:
            result.append(line)
    json_result = "\n".join(result)
    try:
        return json.loads(json_result)
    except:
        print("Parse failed: cannot parse")
        return None

def parse_triplets(output):
    json_result = parse_json(output)

    if json_result is not None:
        result = []
        try:
            triplets = json_result
            if type(triplets) == list:
                for triplet in triplets:
                    if type(triplet) == list and len(triplet) == 3:
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

def get_model(few_shot=True):
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
        
    generate_few_shot_examples = [
        {
            'text': 'Ahmet Ertegun, from the United States, comes from a non performing personnel background and his genre is rhythm and blues. Blues music gave us Rhythm and Blues and Rhythm and Blues gave us Disco',
            'output': '```json\n[["Ahmet_Ertegun", "background", "\"non_performing_personnel\""],\n["Ahmet_Ertegun", "genre", "Rhythm_and_blues"],\n["Ahmet_Ertegun", "country", "United_States"],\n["Rhythm_and_blues", "derived_from", "Blues_music"],\n["Disco", "derived_from", "Rhythm_and_blues"]]\n```',
        },
        {
            'text': 'The 17068.8 millimeter long ALCO RS-3 has a diesel-electric transmission.',
            'output': '```json\n[["ALCO_RS-3", "powerType", "Diesel-electric_transmission"],\n["ALCO_RS-3", "length", "17068.8_(millimetres)"]]\n```',
        },
        {
            'text': 'Alan Shepard was an American who was born on Nov 18, 1923 in New Hampshire, was selected by NASA in 1959, was a member of the Apollo 14 crew and died in California',
            'output': '```json\n[[\"Alan_Shepard\", \"bornIn\", \"New_Hampshire\"],\n[\"Alan_Shepard\", \"diedIn\", \"California\"],\n[\"Alan_Shepard\", \"selectedBy\", \"NASA\"],\n[\"Alan_Shepard\", \"crewMember\", \"Apollo_14\"],\n[\"Alan_Shepard\", \"nationality\", \"American\"],\n[\"Alan_Shepard\", \"bornOn\", \"Nov_18_1923\"]]\n```',
        },
        {
            "text": "The Great Wall of China, built between the 7th century BC and the 16th century, is a series of fortifications made of stone, brick, tamped earth, wood, and other materials. It was constructed to protect the Chinese states and empires against various nomadic groups from the Eurasian Steppe. The wall stretches from Dandong in the east to Lop Lake in the west, along an arc that roughly delineates the southern edge of Inner Mongolia. The most well-known sections of the wall were built by the Ming dynasty (1368–1644).",
            "output": "```json\n[[\"Great_Wall_of_China\", \"builtBetween\", \"7th_century_BC\"],\n[\"Great_Wall_of_China\", \"builtBetween\", \"16th_century\"],\n[\"Great_Wall_of_China\", \"material\", \"stone\"],\n[\"Great_Wall_of_China\", \"material\", \"brick\"],\n[\"Great_Wall_of_China\", \"material\", \"tamped_earth\"],\n[\"Great_Wall_of_China\", \"material\", \"wood\"],\n[\"Great_Wall_of_China\", \"constructedFor\", \"protecting_Chinese_states\"],\n[\"Great_Wall_of_China\", \"against\", \"nomadic_groups\"],\n[\"Great_Wall_of_China\", \"stretchesFrom\", \"Dandong\"],\n[\"Great_Wall_of_China\", \"stretchesTo\", \"Lop_Lake\"],\n[\"Great_Wall_of_China\", \"delineates\", \"southern_edge_of_Inner_Mongolia\"],\n[\"Great_Wall_of_China\", \"wellKnownSectionsBuiltBy\", \"Ming_dynasty\"]]\n```"
        },
        {
            "text": "Isaac Newton, born on January 4, 1643, in Woolsthorpe, England, was a mathematician, physicist, astronomer, and author who is widely recognized as one of the most influential scientists of all time. He formulated the laws of motion and universal gravitation, which laid the foundation for classical mechanics. His book 'Philosophiæ Naturalis Principia Mathematica', published in 1687, is considered one of the most important works in the history of science. Newton made significant contributions to optics and shares credit with Gottfried Wilhelm Leibniz for the development of calculus.",
            "output": "```json\n[[\"Isaac_Newton\", \"bornOn\", \"January_4_1643\"],\n[\"Isaac_Newton\", \"bornIn\", \"Woolsthorpe_England\"],\n[\"Isaac_Newton\", \"profession\", \"mathematician\"],\n[\"Isaac_Newton\", \"profession\", \"physicist\"],\n[\"Isaac_Newton\", \"profession\", \"astronomer\"],\n[\"Isaac_Newton\", \"profession\", \"author\"],\n[\"Isaac_Newton\", \"recognizedAs\", \"influential_scientist\"],\n[\"Isaac_Newton\", \"formulated\", \"laws_of_motion\"],\n[\"Isaac_Newton\", \"formulated\", \"universal_gravitation\"],\n[\"Isaac_Newton\", \"laidFoundationFor\", \"classical_mechanics\"],\n[\"Isaac_Newton\", \"published\", \"Philosophiæ_Naturalis_Principia_Mathematica_1687\"],\n[\"Philosophiæ_Naturalis_Principia_Mathematica\", \"considered\", \"important_work_in_science\"],\n[\"Isaac_Newton\", \"contributedTo\", \"optics\"],\n[\"Isaac_Newton\", \"sharesCreditWith\", \"Gottfried_Wilhelm_Leibniz\"],\n[\"Isaac_Newton\", \"developed\", \"calculus\"]]\n```"
        }
    ]

    generate_instruction = "Given a piece of text, extract relational triplets in the form of [Subject, Relation, Object] from it. Subject and Object should use snake_case. Relation should not contain any spaces and should use camelCase. Relation should be concise, usually consiting of only one word.\nNow please extract triplets from the following text and give the triplets in the same format (as json array of triplets) in a json block :\n{text}"
    generate_few_shot = ChatPromptTemplate.from_messages(
        [
            ("human", generate_instruction),
            ("ai", "{output}")
        ]
    )

    generate_few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=generate_few_shot,
        examples=generate_few_shot_examples,
    )
        
    generate_prompt_template = ChatPromptTemplate.from_messages(
        [
            *([generate_few_shot_prompt] if few_shot else []),
            ("human", generate_instruction)
        ]
    )

    generate_chain = ChatChain(generate_prompt_template)

    def generate_triplets(text: str):
        generated_response = generate_chain.invoke({"text": text})
        log_debug("generated_response", generated_response)
        parsed_response = parse_triplets(generated_response)
        log_debug("parsed_response", parsed_response)

        return parsed_response
    
    describe_few_shot_examples = [
        {
            "text": "Alan Shepard was an American who was born on Nov 18, 1923 in New Hampshire, was selected by NASA in 1959, was a member of the Apollo 14 crew and died in California",
            "triplets": "[[\"Alan_Shepard\", \"bornIn\", \"New_Hampshire\"], [\"Alan_Shepard\", \"diedIn\", \"California\"], [\"Alan_Shepard\", \"selectedBy\", \"NASA\"], [\"Alan_Shepard\", \"crewMember\", \"Apollo_14\"], [\"Alan_Shepard\", \"nationality\", \"American\"], [\"Alan_Shepard\", \"bornOn\", \"Nov_18_1923\"]]",
            "output": "```json\n{\n  \"bornOn\": \"The subject entity was born on the date specified by the object entity.\",\n  \"bornIn\": \"The subject entity was born in the location specified by the object entity.\",\n  \"selectedBy\": \"The subject entity was selected by the object entity.\",\n" +\
                      "  \"crewMember\": \"The subject entity was a member of the crew specified by the object entity.\",\n  \"nationality\": \"The subject entity has nationality specified by the object entity.\",\n  \"diedIn\": \"The subject entity died in the location specified by the object entity.\"\n}\n```"
        },
        {
            "text": "Thomas Edison, an American inventor, was born on February 11, 1847, in Milan, Ohio. He developed many devices, including the phonograph, the motion picture camera, and the electric light bulb.",
            "triplets": "[[\"Thomas_Edison\", \"nationality\", \"American\"], [\"Thomas_Edison\", \"bornOn\", \"February_11_1847\"], [\"Thomas_Edison\", \"bornIn\", \"Milan_Ohio\"], [\"Thomas_Edison\", \"invented\", \"phonograph\"], [\"Thomas_Edison\", \"invented\", \"motion_picture_camera\"], [\"Thomas_Edison\", \"invented\", \"electric_light_bulb\"]]",
            "output": "```json\n{\n  \"nationality\": \"The subject entity has the nationality specified by the object entity.\",\n  \"bornOn\": \"The subject entity was born on the date specified by the object entity.\",\n  \"bornIn\": \"The subject entity was born in the location specified by the object entity.\",\n  \"invented\": \"The subject entity invented the device specified by the object entity.\"\n}\n```"
        },
        {
            "text": "Photosynthesis is a process used by plants and other organisms to convert light energy into chemical energy that can later be released to fuel the organism's activities. This process involves the intake of carbon dioxide and water, which are then converted into glucose and oxygen.",
            "triplets": "[[\"Photosynthesis\", \"usedBy\", \"plants\"], [\"Photosynthesis\", \"usedBy\", \"other_organisms\"], [\"Photosynthesis\", \"converts\", \"light_energy\"], [\"Photosynthesis\", \"convertsTo\", \"chemical_energy\"], [\"Photosynthesis\", \"intakes\", \"carbon_dioxide\"], [\"Photosynthesis\", \"produces\", \"glucose\"], [\"Photosynthesis\", \"produces\", \"oxygen\"]]",
            "output": "```json\n{\n  \"usedBy\": \"The subject entity is used by the entities specified by the object entity.\",\n  \"converts\": \"The subject entity converts the input specified by the object entity into another form.\",\n  \"convertsTo\": \"The subject entity converts an input into the form specified by the object entity.\",\n  \"intakes\": \"The subject entity takes in the input specified by the object entity.\",\n  \"produces\": \"The subject entity produces the output specified by the object entity.\"\n}\n```"
        }
    ]

    describe_instruction = "Given a piece of text and a list of relational triplets extracted from it, write a definition for each relation present.\nText:\n{text}\n\nTriplets:\n{triplets}"
    describe_few_shot = ChatPromptTemplate.from_messages(
        [
            ("human", describe_instruction),
            ("ai", "{output}")
        ]
    )

    describe_few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=describe_few_shot,
        examples=describe_few_shot_examples,
    )

    describe_prompt_template = ChatPromptTemplate.from_messages(
        [
            *([describe_few_shot_prompt] if few_shot else []),
            ("human", describe_instruction)
        ]
    )

    describe_chain = ChatChain(describe_prompt_template)

    def parse_descriptions(output, triplets):
        json_result = parse_json(output)

        if json_result is not None:
            result = {}
            try:
                descriptions = json_result
                if type(descriptions) == dict:
                    for triplet in triplets:
                        relation = triplet[1]
                        if relation in descriptions:
                            result[relation] = descriptions[relation]
                    return result
                else:
                    print("Parse failed: Not a dict")
                    return None
            except:
                print("Parse failed: cannot parse")
                return None

    def describe_triplets(text: str, triplets: List[List[str]]):
        if triplets is None or len(triplets) == 0:
            return None
        generated_response = describe_chain.invoke({"text": text, "triplets": triplets})
        log_debug("generated_response", generated_response)
        parsed_response = parse_descriptions(generated_response, triplets)
        log_debug("parsed_response", parsed_response)

        return parsed_response
    
    return generate_triplets, describe_triplets