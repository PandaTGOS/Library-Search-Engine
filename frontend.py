import pickle
from flask import Flask, render_template, request
import MySQLdb
from MySQLdb.cursors import DictCursor
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask import flash, redirect, url_for
from werkzeug.security import generate_password_hash, check_password_hash
import pandas as pd
import spacy
import string
import gensim
import operator
import re
from gensim import corpora
from gensim.similarities import MatrixSimilarity
from operator import itemgetter
import unicodedata

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key_here'
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = '2006'
app.config['MYSQL_DB'] = 'auth'

login_manager = LoginManager(app)
login_manager.login_view = 'login'

def get_db_connection():
    return MySQLdb.connect(
        host=app.config['MYSQL_HOST'],
        user=app.config['MYSQL_USER'],
        passwd=app.config['MYSQL_PASSWORD'],
        db=app.config['MYSQL_DB'],
        cursorclass=DictCursor
    )
    
class User(UserMixin):
    def __init__(self, id, username, email):
        self.id = id
        self.username = username
        self.email = email

    @staticmethod
    def get(user_id):
        conn = get_db_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute("SELECT * FROM users WHERE id = %s", (user_id,))
                user = cursor.fetchone()
                if user:
                    return User(id=user['id'], username=user['username'], email=user['email'])
        finally:
            conn.close()
        return None

@login_manager.user_loader
def load_user(user_id):
    return User.get(user_id)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        conn = get_db_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
                user = cursor.fetchone()
                if user and check_password_hash(user['password_hash'], password):
                    user_obj = User(id=user['id'], username=user['username'], email=user['email'])
                    login_user(user_obj)
                    return redirect(url_for('index'))
                flash('Invalid username or password')
        except MySQLdb.Error as e:
            flash(f'An error occurred: {str(e)}')
        finally:
            conn.close()
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        
        conn = get_db_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute("SELECT * FROM users WHERE username = %s OR email = %s", (username, email))
                if cursor.fetchone():
                    flash('Username or email already exists.')
                    return render_template('register.html')
                
                hashed_password = generate_password_hash(password)
                cursor.execute("INSERT INTO users (username, email, password_hash) VALUES (%s, %s, %s)",
                               (username, email, hashed_password))
                conn.commit()
                flash('Registration successful. Please log in.')
                return redirect(url_for('login'))
        except MySQLdb.Error as e:
            conn.rollback()
            flash(f'An error occurred: {str(e)}')
        finally:
            conn.close()
    return render_template('register.html')

# Load dataset
dataset = "Book_Dataset_1.csv"
df_books = pd.read_csv(dataset)

# Remove unnecessary columns
columns_to_remove = ['Price', 'Price_After_Tax', 'Tax_amount', 'Avilability', 'Number_of_reviews']
df_books = df_books.drop(columns=columns_to_remove)

# Load stop words
spacy_nlp = spacy.load('en_core_web_sm')
punctuations = string.punctuation
stop_words = spacy.lang.en.stop_words.STOP_WORDS

def spacy_tokenizer(sentence):
    #Normalize to NFC - handle non-ASCII characters better
    sentence = unicodedata.normalize("NFC", sentence)
    
    #optimized regex patterns
    sentence = re.sub(r"[‘’`]", "'", sentence) 
    sentence = re.sub(r"\w*\d\w*", "", sentence) 
    sentence = re.sub(r" +", " ", sentence.strip())  
    sentence = re.sub(r"\n+", " ", sentence) 
    sentence = re.sub(r"[^\w\s.,!?]", " ", sentence) 
    
    tokens = spacy_nlp(sentence)
    tokens = [
        word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ 
        for word in tokens
    ]
    
    tokens = [
        word for word in tokens 
        if word not in stop_words 
        and word not in punctuations 
        and len(word) > 2 
        and not word.isspace()
    ]
    
    return tokens

# Create tokenized description column
df_books['Book_Description_tokenized'] = df_books['Book_Description'].map(lambda x: spacy_tokenizer(x))

# Load pre-trained models or train models if necessary
try:
    with open('models.pickle', 'rb') as f:
        book_tfidf_model, book_lsi_model, dictionary = pickle.load(f)
except FileNotFoundError:
    # Create and train TF-IDF model
    dictionary = corpora.Dictionary(df_books['Book_Description_tokenized'])
    corpus = [dictionary.doc2bow(desc) for desc in df_books['Book_Description_tokenized']]
    book_tfidf_model = gensim.models.TfidfModel(corpus, id2word=dictionary)

    # Create and train LSI model
    book_lsi_model = gensim.models.LsiModel(book_tfidf_model[corpus], id2word=dictionary, num_topics=300)

    # Save models to pickle file
    with open('models.pickle', 'wb') as f:
        pickle.dump((book_tfidf_model, book_lsi_model, dictionary), f)


# Load indexed corpus
book_tfidf_corpus = gensim.corpora.MmCorpus('book_tfidf_model_mm')
book_lsi_corpus = gensim.corpora.MmCorpus('book_lsi_model_mm')
book_index = MatrixSimilarity(book_lsi_corpus, num_features = book_lsi_corpus.num_terms)

@app.route('/')
@login_required
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
@login_required
def search():
    query = request.form['query']
    results = search_similar_books(query, dictionary)
    return render_template('results.html', results=results)

def search_similar_books(search_term, dictionary):
    query_bow = dictionary.doc2bow(spacy_tokenizer(search_term))
    query_tfidf = book_tfidf_model[query_bow]
    query_lsi = book_lsi_model[query_tfidf]

    book_index.num_best = 5

    books_list = book_index[query_lsi]

    books_list.sort(key=itemgetter(1), reverse=True)
    book_names = []

    for j, book in enumerate(books_list):
        # Truncate the book description to the first three sentences
        description = df_books['Book_Description'][book[0]]
        sentences = re.split(r'(?<=[.!?])\s+', description)[:3]  # Split sentences
        truncated_description = ' '.join(sentences)

        book_names.append({
            'Relevance': round((book[1] * 100),2),
            'book Title': df_books['Title'][book[0]],
            'book Plot': truncated_description,
            'Image_Link': df_books['Image_Link'][book[0]]
        })

        if j == (book_index.num_best-1):
            break

    return book_names

if __name__ == '__main__':
    app.run(debug=True)