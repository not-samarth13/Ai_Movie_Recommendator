import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class AIRecommenderApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🎬 AI Movie Recommender")
        self.root.geometry("900x550")
        
        
        self.bg_color = "#C876C6"      
        self.accent_color = "#2B7A78"  
        self.text_color = "#17252A"    
        
        self.root.configure(bg=self.bg_color)
        
        
        style = ttk.Style()
        style.theme_use('clam') 
        
        
        style.configure(
            "Accent.TButton", 
            font=("Segoe UI", 11, "bold"), 
            foreground="white", 
            background=self.accent_color, 
            padding=10
        )
        style.map("Accent.TButton", background=[('active', '#3AAFA9')])
        
        #
        style.configure("TFrame", background=self.bg_color)
        style.configure("TLabel", background=self.bg_color, foreground=self.text_color)
        
       
        self.setup_ai_model()
        
        
        main_frame = ttk.Frame(root, padding=30)
        main_frame.pack(fill=tk.BOTH, expand=True)

        
        ttk.Label(
            main_frame, 
            text="What are you in the mood to watch?", 
            font=("Segoe UI", 18, "bold")
        ).pack(pady=(0, 5))

        
        ttk.Label(
            main_frame, 
            text="Describe the vibe (e.g., 'A tragic romance' or 'space travel')", 
            font=("Segoe UI", 11, "italic"),
            foreground="#DE1D7D"
        ).pack(pady=(0, 20))

        
        self.user_input = tk.Entry(
            main_frame, 
            width=45, 
            font=("Segoe UI", 13), 
            relief=tk.FLAT,
            bg="white",
            fg=self.text_color,
            highlightthickness=1,
            highlightbackground="#CCCCCC",
            highlightcolor=self.accent_color
        )
        self.user_input.pack(pady=10, ipady=6) 
        self.root.bind('<Return>', lambda event: self.get_recommendations())

        
        ttk.Button(
            main_frame, 
            text=" Get Recommendations", 
            style="Accent.TButton",
            command=self.get_recommendations
        ).pack(pady=15)

        
        result_frame = tk.Frame(main_frame, bg="#E0E0E0", bd=1)
        result_frame.pack(pady=10, fill=tk.BOTH, expand=True)

     
        self.result_text = tk.Text(
            result_frame, 
            font=("Segoe UI", 11), 
            state=tk.DISABLED,
            bg="#FFFFFF",
            fg=self.text_color,
            relief=tk.FLAT,
            padx=20,
            pady=20,
            wrap=tk.WORD
        )
        self.result_text.pack(fill=tk.BOTH, expand=True)
        
        
        self.result_text.tag_config("title", font=("Segoe UI", 13, "bold"), foreground=self.accent_color)
        self.result_text.tag_config("match", font=("Segoe UI", 10, "bold"), foreground="#D9534F") # Red for percentage
        self.result_text.tag_config("divider", foreground="#E0E0E0")

    def setup_ai_model(self):
        """Loads the database and trains the TF-IDF vectorizer."""
        data = {
            "Item": ["Inception", "The Matrix", "Titanic", "The Notebook", "Interstellar", "Mad Max: Fury Road","Animal","The Real Jckpot","Tumbbad","Chichhore",
                     "The Godfather","The Dark Knight","Pulp Fiction","Forrest Gump","The Shawshank Redemption","Jurassic Park","The Silence of the Lambs","Back to the Future","Avengers: Endgame",],
            "Description": [
                "A mind-bending sci-fi action movie about dream heists.",
                "A dystopian sci-fi action film exploring virtual reality and rebellion.",
                "A tragic romance and historical drama set on a sinking ship.",
                "A deeply emotional romance and drama about a lifelong love story.",
                "A dramatic sci-fi epic about space travel, black holes, and time.",
                "A high-octane post-apocalyptic action movie set in a dystopian desert.",
                "A Movie Based On Father-Son Relationship And An Action Packed Movie",
                "A Treasure Hunt ANd Thrill Movie",
                "A dark, visually stunning mythological horror-fantasy film exploring a family's destructive greed across generations over a hidden, cursed treasure of a forgotten god.",
                "A nostalgic college comedy-drama that toggles between past university life and a present-day crisis, teaching that failure is just as important as winning.", 
                "A gripping crime drama about the aging patriarch of an organized crime mafia dynasty transferring control of his clandestine empire to his reluctant son.",
                "A gritty superhero action thriller where Batman faces off against his psychological nemesis, the Joker, who wants to plunge Gotham City into anarchy.",
                "A stylized, non-linear crime movie blending dark comedy and violence, intertwining the lives of mob hitmen, a boxer, and a gangster's wife.",
                "A heartwarming historical drama following a slow-witted but kind man who inadvertently influences several defining historical events in the 20th-century United States.",
                "A powerful prison drama about a banker wrongfully convicted of murder who finds hope and redemption over two decades through a lifelong friendship.",
                "A thrilling sci-fi adventure where a theme park populated by cloned dinosaurs suffers a massive security breakdown, trapping a group of visitors.",
                "A tense psychological horror thriller following a young FBI cadet who must confide in an incarcerated, manipulative cannibal killer to catch another serial killer.",
                "A classic sci-fi comedy about a teenager who is accidentally sent thirty years into the past in a time-traveling car and must make his high-school-aged parents fall in love.",
                "An epic superhero action spectacle where the surviving heroes unite for a final time-traveling heist to reverse a universal tragedy.",
            ]
        }
        self.df = pd.DataFrame(data)
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df["Description"])

    def get_recommendations(self):
        """Processes user input and updates the UI with recommendations."""
        query = self.user_input.get().strip()
        
        if not query:
            messagebox.showwarning("Empty Input", "Please type what you are in the mood for.")
            return

        user_vector = self.vectorizer.transform([query])
        similarity_scores = cosine_similarity(user_vector, self.tfidf_matrix).flatten()
        self.df["Match_Score"] = similarity_scores
        
        recommendations = self.df[self.df["Match_Score"] > 0].sort_values(by="Match_Score", ascending=False)
        self.display_results(recommendations)

    def display_results(self, recommendations):
        """Clears the old text and displays the new results with styled tags."""
        self.result_text.config(state=tk.NORMAL)
        self.result_text.delete("1.0", tk.END)

        if recommendations.empty:
            self.result_text.insert(tk.END, "No strong matches found. Try describing something else!\n")
        else:
            for index, row in recommendations.iterrows():
                match_percentage = round(row["Match_Score"] * 100, 1)
                
                
                self.result_text.insert(tk.END, f"🎬 {row['Item']}\n", "title")
                self.result_text.insert(tk.END, f"Match: {match_percentage}%\n", "match")
                self.result_text.insert(tk.END, "────────────────────────────────────────\n", "divider")

        self.result_text.config(state=tk.DISABLED)


if __name__ == "__main__":
    root = tk.Tk()
    app = AIRecommenderApp(root)
    root.mainloop()