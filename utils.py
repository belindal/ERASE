from mistral_common.protocol.instruct.messages import (
    UserMessage,
)
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.tokens.instruct.normalize import ChatCompletionRequest
import tiktoken
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast


TOKENIZER = {
    "mistralai/Mixtral-8x7B-Instruct-v0.1": MistralTokenizer.v3(),
    "mistralai/Mixtral-8x22B-Instruct-v0.1": MistralTokenizer.v3(),
    "meta-llama/Llama-3-70b-chat-hf": None,
    "meta-llama/Llama-3-8b-chat-hf": None,
    "gpt-4o": tiktoken.get_encoding("cl100k_base"),
}

def count_tokens(tokenizer, text):
    if isinstance(tokenizer, MistralTokenizer):
        tokens = tokenizer.encode_chat_completion(ChatCompletionRequest(
            messages=[
                UserMessage(content=text),
            ],
        )).tokens
    elif isinstance(tokenizer, PreTrainedTokenizer) or isinstance(tokenizer, PreTrainedTokenizerFast):
        tokens = tokenizer.tokenize(text)
    elif isinstance(tokenizer, tiktoken.core.Encoding):
        tokens = tokenizer.encode(text)
    return len(tokens)