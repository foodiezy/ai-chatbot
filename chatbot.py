import aiml
import re
import os
import requests
from similarity import SimilarityQA
from logic import LogicEngine
from vision import load_vision_model, classify_user_image

class ChatBot:
    def __init__(self):
        print("Initializing Companion Animals Chatbot...")
        print("Loading AIML Kernel...")
        self.kernel = aiml.Kernel()
        self.kernel.verbose(isVerbose=False)
        self.kernel.learn("animals.aiml")
        
        print("Loading Similarity QA System...")
        self.qa_system = SimilarityQA("animals_qa.csv")
        
        print("Loading Knowledge Base Engine...")
        self.logic_engine = LogicEngine("kb.txt")
        
        print("Loading Cloud Engine...")

        
        print("Loading Image Classification Model...")
        self.vision_model = load_vision_model()
        if not self.vision_model:
            print("Warning: Vision model not found. Some features will be unavailable.")

    def reply(self, text):
        print(f"Bot: {text}")

    def run(self):
        while True:
            try:
                user_input = input("You: ")
                if not user_input.strip():
                    continue

                if user_input.lower() in ['quit', 'exit', 'bye']:
                    self.reply("Goodbye!")
                    break


                multi_learn_match = re.search(r'^i know that (.+) ([a-z]+) (.+)$', user_input.lower())
                learn_match = re.search(r'^i know that (.+) is (.+)$', user_input.lower())
                
                if learn_match:
                    subject = learn_match.group(1).strip()
                    prop = learn_match.group(2).strip()
                    response = self.logic_engine.add_knowledge(subject, prop)
                    self.reply(response)
                    continue
                elif multi_learn_match:
                    subject = multi_learn_match.group(1).strip()
                    verb = multi_learn_match.group(2).strip()
                    obj = multi_learn_match.group(3).strip()
                    if verb == "is":
                        pass
                    else:
                        response = self.logic_engine.add_multivalued(subject, verb, obj)
                        self.reply(response)
                        continue


                multi_check_match = re.search(r'^check that (.+) ([a-z]+) (.+)$', user_input.lower())
                check_match = re.search(r'^check that (.+) is (.+)$', user_input.lower())
                
                if check_match:
                    subject = check_match.group(1).strip()
                    prop = check_match.group(2).strip()
                    response = self.logic_engine.check_knowledge(subject, prop)
                    self.reply(response)
                    continue
                elif multi_check_match:
                    subject = multi_check_match.group(1).strip()
                    verb = multi_check_match.group(2).strip()
                    obj = multi_check_match.group(3).strip()
                    if verb != "is":
                        response = self.logic_engine.check_multivalued(subject, verb, obj)
                        self.reply(response)
                        continue


                if "image" in user_input.lower() or "picture" in user_input.lower():
                    self.reply("Please select an image in the file dialog.")
                    response = classify_user_image(self.vision_model)
                    self.reply(response)
                    continue


                wiki_match = re.search(r'^wiki (.+)', user_input.lower())
                if wiki_match:
                    query = wiki_match.group(1).strip()
                    try:
                        # Format query for Wikipedia (replace spaces with underscores, title case often works best)
                        formatted_query = query.replace(' ', '_').title()
                        api_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{formatted_query}"
                        
                        # Professional headers to avoid 403 Forbidden blocks
                        headers = {
                            'User-Agent': 'CompanionAnimalsChatbot/1.0 (Contact: administrator@example.com; University Project)',
                            'Accept': 'application/json'
                        }
                        
                        response = requests.get(api_url, headers=headers)
                        
                        if response.status_code == 200:
                            data = response.json()
                            summary = data.get('extract')
                            if summary:
                                # Limit to first 2 sentences
                                sentences = summary.split('. ')
                                final_summary = '. '.join(sentences[:2]).strip()
                                if not final_summary.endswith('.'):
                                    final_summary += '.'
                                self.reply(final_summary)
                                continue

                        # Fallback: Search for the title if direct summary fails
                        search_url = "https://en.wikipedia.org/w/api.php"
                        search_params = {
                            "action": "query",
                            "list": "search",
                            "srsearch": query,
                            "format": "json"
                        }
                        search_response = requests.get(search_url, params=search_params, headers=headers)
                        
                        if search_response.status_code == 200:
                            search_data = search_response.json()
                            results = search_data.get('query', {}).get('search', [])
                            
                            if results:
                                best_match = results[0]['title'].replace(' ', '_')
                                fallback_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{best_match}"
                                fallback_response = requests.get(fallback_url, headers=headers)
                                
                                if fallback_response.status_code == 200:
                                    fallback_data = fallback_response.json()
                                    summary = fallback_data.get('extract')
                                    if summary:
                                        sentences = summary.split('. ')
                                        final_summary = '. '.join(sentences[:2]).strip()
                                        if not final_summary.endswith('.'):
                                            final_summary += '.'
                                        self.reply(final_summary)
                                        continue
                        
                        self.reply("Cloud search found no matching pages.")
                    except Exception as e:
                        self.reply(f"Cloud search failed: {e}")
                    continue


                aiml_response = self.kernel.respond(user_input.upper())
                if aiml_response:
                    self.reply(aiml_response)
                    continue


                similarity_response = self.qa_system.get_best_match(user_input)
                if similarity_response:
                    self.reply(similarity_response)
                    continue


                self.reply("I'm not sure how to answer that yet.")
                
            except KeyboardInterrupt:
                print("\nBot: Goodbye!")
                break
            except Exception as e:
                self.reply(f"Sorry, I ran into an error: {e}")

if __name__ == "__main__":
    bot = ChatBot()
    bot.run()
