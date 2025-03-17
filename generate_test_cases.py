import csv
import random

# ------------------------------
# Movie data: 50 movies (10 per year: 2019-2023)
# ------------------------------
movies = [
    # 2019 movies
    {"title": "Joker", "year": "2019", "director": "Todd Phillips", "lead": "Joaquin Phoenix", "genre": "Crime Drama"},
    {"title": "Avengers Endgame", "year": "2019", "director": "Anthony and Joe Russo", "lead": "Robert Downey Jr.", "genre": "Superhero"},
    {"title": "Uncut Gems", "year": "2019", "director": "Safdie brothers ", "lead": "Adam Sandler", "genre": "Crime Thriller"},
    {"title": "Spider-Man: Far From Home", "year": "2019", "director": "Jon Watts", "lead": "Tom Holland", "genre": "Superhero"},
    {"title": "Ford v Ferrari", "year": "2019", "director": "James Mangold", "lead": "Matt Damon", "genre": "Sports Drama"},
    {"title": "Toy Story 4", "year": "2019", "director": "Josh Cooley", "lead": "Tom Hanks", "genre": "Animation"},
    {"title": "Klaus", "year": "2019", "director": "Sergio Pablos", "lead": "Sergio Pablos", "genre": "Animation"},
    {"title": "Five Feet Apart", "year": "2019", "director": "Justin Baldoni", "lead": "Cole Sprouse", "genre": "Romantic Drama"},
    {"title": "The Irishman", "year": "2019", "director": "Martin Scorsese", "lead": "Robert De Niro", "genre": "Crime"},
    {"title": "The Lighthouse", "year": "2019", "director": "Robert Eggers", "lead": "Willem Dafoe", "genre": "Horror"},
    # 2020 movies
    {"title": "Bad Boys for Life", "year": "2020", "director": "Adil El Arbi and Bilall Fallah", "lead": "Will Smith", "genre": "Action"},
    {"title": "Sonic the Hedgehog", "year": "2020", "director": "Jeff Fowler", "lead": "Ben Schwartz", "genre": "Adventure"},
    {"title": "The Invisible Man", "year": "2020", "director": "Leigh Whannell", "lead": "Elisabeth Moss", "genre": "Horror"},
    {"title": "Extraction", "year": "2020", "director": "Sam Hargrave", "lead": "Chris Hemsworth", "genre": "Action"},
    {"title": "Tenet", "year": "2020", "director": "Christopher Nolan", "lead": "John David Washington", "genre": "Sci-Fi"},
    {"title": "Birds of Prey", "year": "2020", "director": "Cathy Yan", "lead": "Margot Robbie", "genre": "Action"},
    {"title": "Onward", "year": "2020", "director": "Dan Scanlon", "lead": "Tom Holland", "genre": "Animation"},
    {"title": "The Gentlemen", "year": "2020", "director": "Guy Ritchie", "lead": "Matthew McConaughey", "genre": "Crime"},
    {"title": "Mulan", "year": "2020", "director": "Niki Caro", "lead": "Liu Yifei", "genre": "Action"},
    {"title": "The Trial of the Chicago 7", "year": "2020", "director": "Aaron Sorkin", "lead": "Eddie Redmayne", "genre": "Drama"},
    # 2021 movies
    {"title": "The Suicide Squad", "year": "2021", "director": "James Gunn", "lead": "Margot Robbie", "genre": "Action"},
    {"title": "Black Widow", "year": "2021", "director": "Cate Shortland", "lead": "Scarlett Johansson", "genre": "Superhero"},
    {"title": "Jungle Cruise", "year": "2021", "director": "Jaume Collet-Serra", "lead": "Emily Blunt", "genre": "Adventure"},
    {"title": "No Time to Die", "year": "2021", "director": "Cary Joji Fukunaga", "lead": "Daniel Craig", "genre": "Action"},
    {"title": "Dune", "year": "2021", "director": "Denis Villeneuve", "lead": "Timothée Chalamet", "genre": "Sci-Fi"},
    {"title": "Free Guy", "year": "2021", "director": "Shawn Levy", "lead": "Ryan Reynolds", "genre": "Action Comedy"},
    {"title": "Space Jam: A New Legacy", "year": "2021", "director": "Malcolm D. Lee", "lead": "LeBron James", "genre": "Animation"},
    {"title": "Venom: Let There Be Carnage", "year": "2021", "director": "Andy Serkis", "lead": "Tom Hardy", "genre": "Superhero"},
    {"title": "Shang-Chi", "year": "2021", "director": "Destin Daniel Cretton", "lead": "Simu Liu", "genre": "Superhero"},
    {"title": "Eternals", "year": "2021", "director": "Chloé Zhao", "lead": "Gemma Chan", "genre": "Superhero"},
    # 2022 movies
    {"title": "Top Gun: Maverick", "year": "2022", "director": "Joseph Kosinski", "lead": "Tom Cruise", "genre": "Action"},
    {"title": "Everything Everywhere All at Once", "year": "2022", "director": "Daniel Kwan and Daniel Scheinert", "lead": "Michelle Yeoh", "genre": "Sci-Fi"},
    {"title": "Elvis", "year": "2022", "director": "Baz Luhrmann", "lead": "Austin Butler", "genre": "Biographical"},
    {"title": "Ambulance", "year": "2022", "director": "Michael Bay", "lead": "Jake Gyllenhaal", "genre": "Action"},
    {"title": "The Batman", "year": "2022", "director": "Matt Reeves", "lead": "Robert Pattinson", "genre": "Superhero"},
    {"title": "Avatar: The Way of Water", "year": "2022", "director": "James Cameron", "lead": "Sam Worthington", "genre": "Sci-Fi"},
    {"title": "The Woman King", "year": "2022", "director": "Gina Prince-Bythewood", "lead": "Viola Davis", "genre": "Historical"},
    {"title": "Black Panther: Wakanda Forever", "year": "2022", "director": "Ryan Coogler", "lead": "Letitia Wright", "genre": "Superhero"},
    {"title": "Bullet Train", "year": "2022", "director": "David Leitch", "lead": "Brad Pitt", "genre": "Action"},
    {"title": "The Northman", "year": "2022", "director": "Robert Eggers", "lead": "Alexander Skarsgård", "genre": "Historical"},
    # 2023 movies
    {"title": "Oppenheimer", "year": "2023", "director": "Christopher Nolan", "lead": "Cillian Murphy", "genre": "Historical"},
    {"title": "Barbie", "year": "2023", "director": "Greta Gerwig", "lead": "Margot Robbie", "genre": "Comedy"},
    {"title": "The Flash", "year": "2023", "director": "Andy Muschietti", "lead": "Ezra Miller", "genre": "Superhero"},
    {"title": "Guardians of the Galaxy Vol. 3", "year": "2023", "director": "James Gunn", "lead": "Chris Pratt", "genre": "Superhero"},
    {"title": "Mission: Impossible – Dead Reckoning Part One", "year": "2023", "director": "Christopher McQuarrie", "lead": "Tom Cruise", "genre": "Action"},
    {"title": "Spider-Man: Across the Spider-Verse", "year": "2023", "director": "Jared Bush", "lead": "Shameik Moore", "genre": "Animation"},
    {"title": "Dune: Part Two", "year": "2023", "director": "Denis Villeneuve", "lead": "Timothée Chalamet", "genre": "Sci-Fi"},
    {"title": "John Wick: Chapter 4", "year": "2023", "director": "Chad Stahelski", "lead": "Keanu Reeves", "genre": "Action"},
    {"title": "Indiana Jones and the Dial of Destiny", "year": "2023", "director": "James Mangold", "lead": "Harrison Ford", "genre": "Adventure"},
    {"title": "Elemental", "year": "2023", "director": "Peter Sohn", "lead": "Mae Whitman", "genre": "Animation"}
]

# ------------------------------
# Define question templates for each category
# ------------------------------

# Complex question templates (30 templates)
complex_templates = [
    "How does the directorial vision of {director} in {movie} ({year}) influence its portrayal of {genre} themes?",
    "In what ways does {movie}, directed by {director} and starring {lead}, challenge conventional {genre} tropes?",
    "Discuss the impact of {movie}'s narrative structure on American {genre} cinema in {year}.",
    "How does {movie} compare to other {genre} films in terms of storytelling and character development?",
    "What innovative cinematic techniques does {director} employ in {movie} to enhance its {genre} appeal?",
    "Analyze how the performance of {lead} in {movie} contributes to its commentary on contemporary American culture.",
    "What role does the musical score play in shaping the atmosphere of {movie} ({year})?",
    "Examine the use of visual symbolism in {movie} and its reflection of the cultural zeitgeist of {year}.",
    "How does {movie} reflect the socio-political climate of America in {year} while staying true to {genre} conventions?",
    "In {movie}, how are the themes of ambition and redemption explored through the lens of {genre} storytelling?",
    "What sets {movie} apart from other {genre} films released in {year}?",
    "How do the directorial choices in {movie} enhance the narrative complexity typical of {genre} cinema?",
    "How does {movie} utilize unconventional narrative techniques to redefine the {genre} genre?",
    "In what ways does {movie} innovate in its portrayal of character dynamics and conflict?",
    "Discuss how the collaboration between {director} and {lead} in {movie} elevates its {genre} elements.",
    "What underlying social commentaries are presented in {movie} and how do they relate to the American context of {year}?",
    "How does the cinematography in {movie} reinforce its central themes and {genre} aesthetics?",
    "Examine the critical reception of {movie} and its influence on modern trends in {genre} cinema.",
    "What narrative risks does {movie} take that challenge traditional filmmaking in {year}?",
    "How does {movie} blend humor and drama to comment on contemporary American society?",
    "Discuss the interplay between visual effects and storytelling in {movie}.",
    "What lessons in character development can be learned from {lead}'s performance in {movie}?",
    "How does the editing style of {movie} contribute to its unique narrative structure in the {genre} genre?",
    "What does {movie} reveal about the evolution of American {genre} films in recent years?",
    "How do the film’s thematic elements in {movie} align with the historical context of {year}?",
    "Examine the role of supporting characters in {movie} and their contribution to the overall story.",
    "How is tension built throughout {movie} and what does it suggest about its underlying themes?",
    "Discuss the significance of the film’s climax in shaping the overall message of {movie}.",
    "In what ways does {movie} challenge audience expectations of {genre} cinema?",
    "How does the narrative innovation in {movie} reflect broader trends in American filmmaking?"
]

# Nominal (straightforward) question templates (10 templates)
nominal_templates = [
    "Who directed {movie}?",
    "In which year was {movie} released?",
    "Who starred in {movie}?",
    "What is the genre of {movie}?",
    "Name the lead actor in {movie}.",
    "Identify the director of {movie}.",
    "What year did {movie} come out?",
    "What genre best describes {movie}?",
    "Who is the lead in {movie}?",
    "Which year marks the release of {movie}?"
]

# ------------------------------
# Generate complex questions
# ------------------------------
complex_questions = []
for movie in movies:
    for template in complex_templates:
        question = template.format(
            movie=movie["title"],
            year=movie["year"],
            director=movie["director"],
            lead=movie["lead"],
            genre=movie["genre"]
        )
        complex_questions.append(question)

# We need 500 unique complex questions.
if len(complex_questions) < 500:
    raise ValueError("Not enough complex questions generated.")
complex_sample = random.sample(complex_questions, 500)

# ------------------------------
# Generate nominal questions
# ------------------------------
nominal_questions = []
for movie in movies:
    for template in nominal_templates:
        question = template.format(movie=movie["title"])
        nominal_questions.append(question)

# There should be exactly 50 movies * 10 templates = 500 unique nominal questions.
if len(nominal_questions) != 500:
    raise ValueError("Nominal questions count is not 500; check the templates and movie list.")

# ------------------------------
# Combine both sets, shuffle, and write to CSV
# ------------------------------
all_questions = complex_sample + nominal_questions
random.shuffle(all_questions)

with open("test_cases.csv", "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["question"])  # header
    for question in all_questions:
        writer.writerow([question])

print("test_cases.csv generated with 500 complex and 500 nominal unique questions (total 1000).")
