import openai # type: ignore
import re
import os
class LLMIntegration:

    def __init__(self):
        self.api_key = os.environ.get("OPENAI_API_KEY")
        self.client = openai.OpenAI(api_key=self.api_key, base_url="https://api.groq.com/openai/v1")


    def resolve_task(self, action, elements, model="deepseek-r1-distill-llama-70b"):
        formatted_elements = "\n".join([f"{i+1}. {el}" for i, el in enumerate(elements)])
        prompt = f"""Tu tarea es decidir cuál elemento debe ser pulsado para cumplir la acción del usuario.
    Acción:
    - {action}

    Elementos detectados:
    {formatted_elements}

    Responde **únicamente con el número del elemento más adecuado** para cumplir la acción. No expliques nada más. Solo responde con un número.
    """

        response = self.client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "Eres un asistente que decide qué elemento debe pulsarse en una interfaz gráfica basada en una acción solicitada. Solo debes responder con un número."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.0
        )

        content = response.choices[0].message.content.replace("\n", " ").strip()

        # Buscar el número que aparece justo después de </think>
        match = re.search(r"</think>\s*(\d+)", content, re.IGNORECASE)
        if match:
            return int(match.group(1))

        # Si no hay etiqueta <think>, intentar capturar el único número
        fallback_match = re.search(r"\b(\d+)\b", content)
        if fallback_match:
            return int(fallback_match.group(1))

        raise ValueError("No se encontró un número válido en la respuesta del modelo.")
