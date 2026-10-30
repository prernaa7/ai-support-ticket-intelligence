
import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY not found. Check your .env file.")

client = genai.Client(
    api_key=api_key,
    http_options=types.HttpOptions(timeout=120000)
)


def summarize_ticket(ticket_description):

    prompt = f"""
    You are a customer support ticket assistant.

    Summarize this ticket in 2-3 clear sentences.

    Include:
    - The customer's main issue
    - Important details
    - What help the customer needs

    Do not invent information.

    Ticket:
    {ticket_description}
    """

    try:
        response = client.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents=prompt
        )

        if not response.text:
            return "Gemini returned an empty response."

        return response.text.strip()

    except Exception as error:
        print("Gemini API Error:", type(error).__name__, str(error))
        return "Unable to generate ticket summary."


if __name__ == "__main__":

    ticket = """
    I tried making an online payment of Rs. 5000 through the mobile
    banking application yesterday evening. The amount was deducted
    from my bank account, but the transaction status showed failed.

    I checked my account statement and confirmed that the amount
    has not been credited back. I also contacted customer support
    but have not received any update.

    Please investigate this transaction and help me get my money
    refunded as soon as possible.
    """

    print("CUSTOMER SUPPORT TICKET")
    print("-" * 50)
    print(ticket)

    print("\nGENERATED SUMMARY")
    print("-" * 50)

    summary = summarize_ticket(ticket)
    print(summary)
