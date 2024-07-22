import re
import json

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


text = """```json
[
  ["Transformer", "architecture", "model"],
  ["Transformer", "layer", "multi-head self-attention mechanism"],
  ["Transformer", "layer", "simple, position-wise fully connected feed-forward network"],
  ["Transformer", "layer", "residual connection"],
  ["Transformer", "layer", "layer normalization"],
  ["Transformer", "layer", "output", "dmodel"],
  ["Decoder", "layer", "stack", "6"],
  ["Decoder", "layer", "sub-layer", "multi-head attention over the output of the encoder stack"],
  ["Decoder", "layer", "sub-layer", "residual connection"],
  ["Decoder", "layer", "sub-layer", "layer normalization"],
  ["Decoder", "layer", "self-attention sub-layer", "masking"],
  ["Decoder", "layer", "self-attention sub-layer", "output embeddings offset by one position"],
  ["Attention", "function", "mapping query and a set of key-value pairs to an output"],
  ["Attention", "function", "output", "weighted sum of the values"],
  ["Attention", "function", "compatibility function", "query with the corresponding key"],
  ["Scaled Dot-Product Attention", "input", "queries and keys of dimension dk"],
  ["Scaled Dot-Product Attention", "input", "values of dimension dv"],
  ["Scaled Dot-Product Attention", "output", "weighted sum of the values"],
  ["Multi-Head Attention", "consists of", "several attention layers running in parallel"],
  ["Multi-Head Attention", "output", "matrix Q"],
  ["Multi-Head Attention", "keys and values", "matrices KandV"],
  ["Multi-Head Attention", "compute", "matrix of outputs"],
  ["Dot-product attention", "compute", "compatibility function using a feed-forward network with a single hidden layer"],
  ["Additive attention", "compute", "compatibility function using a feed-forward network with a single hidden layer"],
  ["Dot-product attention", "implementation", "highly optimized matrix multiplication code"]
]
```"""

print(parse_triplets(text))