from flask import Flask, jsonify, render_template, request, redirect, url_for
import psycopg2-binary
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv
import os
app = Flask(__name__)

# Load environment variables from .env file
load_dotenv()
# Access variables
api_key = os.getenv('API_KEY')
db_user = os.getenv('DB_USER')
db_host = os.getenv('DB_HOST')
db_password = os.getenv('DB_PASSWORD')
db_name = os.getenv('DB_NAME')

# Store query-answer pairs temporarily
history = []  

# Initialize SBERT for embedding generation without torch
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')  # Ensure 'cpu' if torch isn't used

# Connect to PostgreSQL Database
def get_db_connection():
    return psycopg2.connect(
        host=db_host,
        database= db_name,
        user= db_user,
        password= db_password
    )

get_db_connection()
# Convert query to embedding and search for the most similar text.
def search_similar_texts(user_query,  table_name, top_n=5):
    
    # Generate the embedding for the user query
    query_embedding = sbert_model.encode(user_query).tolist()
    
    # Connect to the database
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Use pgvector to find the most similar text based on cosine similarity
    query= f"""
        SELECT source, text, embedding <#> %s::vector AS similarity
        FROM {table_name}
        ORDER BY similarity
        LIMIT %s;
    """
    cursor.execute(query, (query_embedding, top_n))
    
    # Fetch the results
    results = cursor.fetchall()
    cursor.close()
    conn.close()
    return results


client = Groq(
    api_key= api_key
)

def final_response(user_query):
    result = search_similar_texts(user_query, "new_data")
    context_info =""
    for x in result:
        context_info = context_info + " " + x[1]
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are an AI assistant. Use the provided context information to craft summarized and readable responses with proper line breaks that are directly relevant to the user's query. Give the answer in bulletpoints wherever need. Use strcictly provided context only to give the response. Format the response in approapriate readable manner with proper spaces and line breaks. Do not add any other info to the response other than provided in the context, also don't show that you are providing the response from a given context. If the user's query's answer is not present in the provided context just tell the user 'Couldn't find the answer'."},
            {"role": "assistant", "content": f"The following context will guide my response: {context_info}"},
            {"role": "user", "content": user_query,}
        ],
        model="llama3-70b-8192",
    )
    return chat_completion.choices[0].message.content

@app.template_filter('format_answer')
def format_answer(answer):
    return answer.replace('\n', '<br>')

@app.route("/", methods=["GET", "POST"])
def query_form():
    if request.method == "POST":
        query = request.form.get("query")
        if query:
            # Generate an answer
            answer = final_response(query)
            
            # Insert query-answer pair into PostgreSQL
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO user_feedback (query, answer)
                VALUES (%s, %s) RETURNING id;
                """,
                (query, answer)
            )
            inserted_id = cursor.fetchone()[0]
            conn.commit()
            cursor.close()
            conn.close()
            history.append({"query": query, "answer": answer, "_id": inserted_id})
            return redirect(url_for("query_form"))  # Redirect to clear POST data
    return render_template("index.html", history=history)

@app.route("/feedback/<int:response_id>/<feedback_type>", methods=["POST"])
def submit_feedback(response_id, feedback_type):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        UPDATE user_feedback
        SET feedback = %s
        WHERE id = %s;
        """,
        (feedback_type, response_id)
    )
    conn.commit()
    cursor.close()
    conn.close()

    return jsonify({"message": "Feedback submitted successfully!"}), 200

if __name__ == "__main__":
    app.run(debug=True)
