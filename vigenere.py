import streamlit as st
from collections import Counter

###################################Rail Fence
#  Chiffrement
def rail_fence_encrypt(text, key):
    rail = [['\n' for _ in range(len(text))] for _ in range(key)]
    direction_down = False
    row, col = 0, 0

    for char in text:
        if row == 0 or row == key - 1:
            direction_down = not direction_down

        rail[row][col] = char
        col += 1

        row += 1 if direction_down else -1

    encrypted_text = ""
    for row in rail:
        encrypted_text += ''.join(char for char in row if char != '\n')
    return encrypted_text

# Déchiffrement
def rail_fence_decrypt(cipher, key):
    rail = [['\n' for _ in range(len(cipher))] for _ in range(key)]
    direction_down = None
    row, col = 0, 0

    for _ in cipher:
        if row == 0:
            direction_down = True
        if row == key - 1:
            direction_down = False

        rail[row][col] = '*'
        col += 1

        row += 1 if direction_down else -1

    index = 0
    for i in range(key):
        for j in range(len(cipher)):
            if rail[i][j] == '*' and index < len(cipher):
                rail[i][j] = cipher[index]
                index += 1

    decrypted_text = ""
    row, col = 0, 0
    for _ in range(len(cipher)):
        if row == 0:
            direction_down = True
        if row == key - 1:
            direction_down = False

        if rail[row][col] != '*':
            decrypted_text += rail[row][col]
            col += 1

        row += 1 if direction_down else -1

    return decrypted_text

############################################## Vigenère 
#  Chiffrement
def vigenere_encrypt(text, key):
    encrypted_text = ""
    key = key.upper()
    key_index = 0

    for char in text:
        if char.isalpha():
            shift = ord(key[key_index]) - ord('A')
            if char.islower():
                encrypted_text += chr((ord(char) - ord('a') + shift) % 26 + ord('a'))
            else:
                encrypted_text += chr((ord(char) - ord('A') + shift) % 26 + ord('A'))

            key_index = (key_index + 1) % len(key)
        else:
            encrypted_text += char

    return encrypted_text

# Déchiffrement
def vigenere_decrypt(text, key):
    decrypted_text = ""
    key = key.upper()
    key_index = 0

    for char in text:
        if char.isalpha():
            shift = ord(key[key_index]) - ord('A')
            if char.islower():
                decrypted_text += chr((ord(char) - ord('a') - shift) % 26 + ord('a'))
            else:
                decrypted_text += chr((ord(char) - ord('A') - shift) % 26 + ord('A'))

            key_index = (key_index + 1) % len(key)
        else:
            decrypted_text += char

    return decrypted_text

##############################################Cryptanalyse 
#_________________________Méthode de Friedman

#FONCTION POUR POUR AVOIR DES SOUS SEQUENCES DU MESSAGE CHIFFRE
def extract_letters_with_step(text, k):
    """
    Cette fonction extrait les lettres d'un texte en utilisant un pas de 1 jusqu'à k.
    
    :param text: Le texte à partir duquel extraire les lettres.
    :param k: Le pas entre les lettres à extraire.
    :return: Une liste contenant les lettres extraites.
    """
    extracted_letters = []
    
    # Parcourir le texte avec un pas de k
    for i in range(0, len(text), k):
        # Ajouter la lettre au résultat si elle est une lettre
        if text[i].isalpha():
            extracted_letters.append(text[i])
    
    return extracted_letters


def calculate_ic_friedman(text):
    """
    Calcule l'Indice de Coïncidence (IC) selon William Friedman pour un texte donné.
    
    :param text: Le texte sur lequel calculer l'IC.
    :return: L'indice de coïncidence.
    """
    # Filtrer pour ne garder que les lettres (A-Z) et les convertir en majuscules
    text = [char.upper() for char in text if char.isalpha()]
    
    n = len(text)  # Longueur du texte (nombre total de lettres)
    if n <= 1:
        return 0  # L'indice de coïncidence n'a pas de sens pour un texte de longueur 1 ou moins
    
    # Compter les occurrences de chaque lettre
    letter_counts = Counter(text)
    
    # Appliquer la formule de l'IC de Friedman
    ic = sum(count * (count - 1) for count in letter_counts.values()) / (n * (n - 1))
    
    return (ic)

def extract_sequences_with_steps_and_calculate_ic(text, m):
    """
    Cette fonction extrait des séquences de lettres dans un texte avec un pas allant de 1 à m
    et calcule l'indice de coïncidence pour chaque séquence.
    
    :param text: Le texte à partir duquel extraire les séquences.
    :param m: Le pas maximum jusqu'auquel extraire les séquences.
    :return: Une liste contenant les séquences extraites et leurs indices de coïncidence.
    """
    sequences_and_ic = []  # Liste pour stocker les séquences et leurs IC
    
    # Pour chaque pas de 1 à m
    for k in range(1, m + 1):
        extracted_sequence = []  # Liste des lettres extraites avec un pas de k
        
        # Parcourir le texte avec un pas de k
        for i in range(0, len(text), k):
            # Ajouter la lettre si elle est une lettre
            if text[i].isalpha():
                extracted_sequence.append(text[i])
        
        # Calculer l'indice de coïncidence pour cette séquence
        ic = calculate_ic_friedman(extracted_sequence)
        
        # Ajouter la séquence et son IC à la liste
        sequences_and_ic.append((extracted_sequence, ic))
    
    return sequences_and_ic

def find_closest_ic_to_target(text, m, target_ic=0.074):
    """
    Cette fonction extrait des séquences de lettres dans un texte avec un pas allant de 1 à m,
    puis trouve la séquence dont l'indice de coïncidence est le plus proche d'une valeur cible (par défaut 0,074).
    
    :param text: Le texte à partir duquel extraire les séquences.
    :param m: Le pas maximum jusqu'auquel extraire les séquences.
    :param target_ic: L'indice de coïncidence cible (par défaut 0,074).
    :return: La séquence et son IC dont l'indice est le plus proche de la valeur cible, ainsi que le pas correspondant.
    """
    closest_sequence = None
    closest_ic = float('inf')  # Initialiser avec un IC très élevé pour trouver le minimum
    closest_diff = float('inf')  # Différence la plus petite
    best_k = -1  # Le pas correspondant à l'IC le plus proche

    # Extraire les séquences avec des pas allant de 1 à m et calculer leur IC
    for k in range(1, m + 1):
        extracted_sequence = []
        
        # Extraire les lettres avec le pas k
        for i in range(0, len(text), k):
            if text[i].isalpha():
                extracted_sequence.append(text[i])

        # Calculer l'IC pour la séquence
        ic = calculate_ic_friedman(extracted_sequence)
        
        # Calculer la différence entre l'IC et la cible
        diff = abs(ic - target_ic)

        # Si cette différence est plus petite que la précédente, mettre à jour
        if diff < closest_diff:
            closest_sequence = extracted_sequence
            closest_ic = ic
            closest_diff = diff
            best_k = k  # Enregistrer le pas correspondant

    return closest_sequence, closest_ic, best_k


 
#FONCTION POUR POUR AVOIR DES SOUS SEQUENCES DU MESSAGE CHIFFRE

def extract_letters_by_modulo_k(text, k):
    """
    Cette fonction extrait les lettres du texte selon les indices qui satisfont P mod k,
    pour P allant de 0 à k-1.
    
    :param text: Le texte dans lequel extraire les lettres.
    :param k: Le nombre modulo utilisé pour extraire les lettres.
    :return: Un dictionnaire où les clés sont les valeurs de P (de 0 à k-1),
             et les valeurs sont les lettres extraites aux indices correspondants.
    """
    # Dictionnaire pour stocker les lettres extraites pour chaque P mod k
    extracted_letters = {p: [] for p in range(k)}
    
    # Parcourir le texte et extraire les lettres selon les indices
    for i in range(len(text)):
        # Vérifier la condition modulo k
        p = i % k
        if text[i].isalpha():  # Filtrer pour ne garder que les lettres
            extracted_letters[p].append(text[i])
    
    return extracted_letters

def calculate_highest_letter_frequency(letters):
    """
    Calcule la fréquence la plus élevée des lettres dans une liste.
    
    :param letters: La liste des lettres dont on veut calculer la fréquence.
    :return: La lettre avec la fréquence la plus élevée et la fréquence correspondante.
    """
    if len(letters) == 0:
        return None, 0.0
    
    letter_counts = Counter(letters)
    most_common_letter, count = letter_counts.most_common(1)[0]  # Lettre la plus fréquente
    highest_frequency = (count / len(letters)) * 100  # Calculer la fréquence en pourcentage
    return most_common_letter, highest_frequency






def solve_equation_modulo(a, b, m=26):
    """
    Résout l'équation a + x ≡ b (mod m) pour x, où m est 26 par défaut.
    
    :param a: Le terme constant de l'équation (dans notre exemple, 4).
    :param b: Le résultat souhaité du côté droit de l'équation (dans notre exemple, 22).
    :param m: Le module, par défaut 26.
    :return: La solution x modulo m.
    """
    # Isoler x : x ≡ (b - a) (mod m)
    x = (b - a) % m
    return x

def text_to_numbers(text):
    """
    Convertit un texte en une liste de nombres, où chaque lettre est remplacée par son index dans l'alphabet.
    A = 0, B = 1, ..., Z = 25
    """
    return [ord(c.upper()) - ord('A') for c in text if c.isalpha()]

def numbers_to_text(numbers):
    """
    Convertit une liste de nombres en texte, en utilisant l'alphabet.
    0 = A, 1 = B, ..., 25 = Z
    """
    return ''.join(chr(num + ord('A')) for num in numbers)


def solve_multiple_equations(extracted_letters):
    
    # Convertir chaque lettre de b en chiffre (0 = A, 1 = B, ..., 25 = Z)
    b_numbers = text_to_numbers(extracted_letters)
    
    # Valeur de a
    a = 4
    
    # Résoudre l'équation pour chaque lettre du texte
    result_numbers = []
    
    for b_num in b_numbers:
        # Résoudre l'équation 4 + x ≡ b_num (mod 26)
        x = solve_equation_modulo(a, b_num)
        if x is not None:
            result_numbers.append(x)
        else:
            result_numbers.append(None)  # Si la solution n'existe pas
    
    # Convertir les résultats en texte
    result_text = numbers_to_text(result_numbers)
    return(result_text)


#________________Méthode de Babbage
def find_repetitions_of_varied_lengths(paragraph, min_block_size):
    """
    Cette fonction détecte toutes les répétitions de blocs de taille 3 ou plus dans un texte.
    
    :param paragraph: Le texte dans lequel rechercher les répétitions.
    :param min_block_size: La taille minimale du bloc à rechercher (ici, 3 caractères).
    :return: Un dictionnaire des blocs et leurs positions dans le texte.
    """
    repetitions = {}  # Dictionnaire pour stocker les blocs et leurs positions
    
    # Parcourir les tailles de blocs de min_block_size à len(paragraph)
    for block_size in range(min_block_size, len(paragraph) + 1):
        # Pour chaque taille de bloc, parcourir le texte pour extraire les sous-blocs
        for i in range(len(paragraph) - block_size + 1):  # On parcourt jusqu'à (len(paragraph) - block_size)
            block = paragraph[i:i + block_size]  # Extraire un sous-bloc de taille block_size
            
            if block not in repetitions:
                repetitions[block] = []  # Si le bloc n'existe pas dans le dictionnaire, l'ajouter
            
            repetitions[block].append(i)  # Ajouter l'indice où le bloc a été trouvé
    
    # Filtrer les blocs qui apparaissent plus d'une fois
    repeated_blocks = {block: positions for block, positions in repetitions.items() if len(positions) > 1}
    
    return repeated_blocks

def calculate_distance_between_repetitions(paragraph, min_block_size):
    """
    Cette fonction calcule la distance entre les deux premières répétitions de chaque bloc
    trouvé par la fonction `find_repetitions_of_varied_lengths`.
    
    :param paragraph: Le texte dans lequel rechercher les répétitions.
    :param min_block_size: La taille minimale du bloc à rechercher.
    :return: Une liste avec chaque bloc, ses positions et les distances entre ces positions.
    """
    
    # Appliquer la fonction pour détecter les blocs répétés
    repeated_blocks = find_repetitions_of_varied_lengths(paragraph, min_block_size)
    
    # Liste pour stocker les résultats formatés
    results = []
    
    # Pour chaque bloc répété, calculer la distance entre les positions
    for block, positions in repeated_blocks.items():
        if len(positions) >= 2:
            # Calculer les distances entre les positions consécutives
            for i in range(len(positions) - 1):
                pos1 = positions[i]
                pos2 = positions[i + 1]
                distance = pos2 - pos1  # Calcul de la distance entre les deux positions
                
                # Ajouter les résultats sous forme d'une ligne bien formatée
                results.append([block, positions, distance])
    
    return results



def prime_factors(n):
    """
    Cette fonction calcule les facteurs premiers uniques de n.
    :param n: Le nombre à décomposer en facteurs premiers.
    :return: Un ensemble contenant les facteurs premiers de n.
    """
    factors = set()  # Utiliser un set pour éviter les répétitions de facteurs premiers
    # Diviser par 2 jusqu'à ce que n ne soit plus divisible par 2
    while n % 2 == 0:
        factors.add(2)
        n //= 2
    # Tester les autres nombres impairs à partir de 3
    for i in range(3, int(n ** 0.5) + 1, 2):
        while n % i == 0:
            factors.add(i)
            n //= i
    # Si n est un nombre premier supérieur à 2
    if n > 2:
        factors.add(n)
    return factors


from collections import Counter
def calculate_distance_and_prime_factors(paragraph, min_block_size):
    """
    Cette fonction calcule la distance entre les répétitions de chaque bloc et les facteurs premiers
    associés à chaque distance.
    
    :param paragraph: Le texte dans lequel rechercher les répétitions.
    :param min_block_size: La taille minimale du bloc à rechercher.
    :return: Une liste avec chaque bloc, ses positions, les distances et les facteurs premiers associés.
    """
    
    # Appliquer la fonction pour détecter les blocs répétés
    repeated_blocks = find_repetitions_of_varied_lengths(paragraph, min_block_size)
    
    # Liste pour stocker les résultats formatés
    results = []
    all_factors = []  # Liste pour stocker tous les facteurs premiers
    
    # Pour chaque bloc répété, calculer la distance entre les positions et les facteurs premiers associés
    for block, positions in repeated_blocks.items():
        if len(positions) >= 2:
            # Calculer les distances entre les positions consécutives
            for i in range(len(positions) - 1):
                pos1 = positions[i]
                pos2 = positions[i + 1]
                distance = pos2 - pos1  # Calcul de la distance entre les deux positions
                
                # Calcul des facteurs premiers de la distance
                factors = prime_factors(distance)
                all_factors.extend(factors)  # Ajouter tous les facteurs premiers à la liste globale
                
                # Ajouter les résultats sous forme d'une ligne bien formatée
                results.append([block, positions, distance, factors])
    
    return results, all_factors


def most_frequent_prime_factors(all_factors):
    """
    Cette fonction calcule les facteurs premiers les plus fréquents par ordre décroissant de fréquence.
    
    :param all_factors: Liste des facteurs premiers collectés pour toutes les distances.
    :return: Liste des facteurs premiers les plus fréquents, triée par fréquence décroissante.
    """
    # Compter la fréquence des facteurs premiers
    factor_counts = Counter(all_factors)
    
    # Trier les facteurs premiers par fréquence décroissante
    sorted_factors = factor_counts.most_common()  # Renvoie une liste de tuples (facteur, fréquence)
    
    return sorted_factors





# Interface Streamlit
st.write("<center><h1>TP CSI</h1></center>", unsafe_allow_html=True)

# Définir les options de choix
options = ["Chiffrement Rail Fence", "Chiffrement Vigenère"]

# Demander à l'utilisateur de choisir une option
menu = st.sidebar.radio("Choisir une méthode:", options)


if menu == "Chiffrement Rail Fence":
    st.write('<center><h1 style="color:#FF6666;">Chiffrement Rail Fence</h1></center>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Veuillez saisir ou importer un texte", type=["txt"])
    # Texte à chiffrer/déchiffrer
    if uploaded_file is not None:
        # Lecture du contenu du fichier texte
        file_content = uploaded_file.read().decode("utf-8")
        st.info("Le contenu du fichier a été pré-rempli. Vous pouvez le modifier.")
    else:
        # Si aucun fichier n'est importé, initialiser un contenu vide
        file_content = ""

    # Champ texte avec le contenu du fichier pré-rempli (ou vide par défaut)
    text = st.text_area("Texte à chiffrer/déchiffrer", file_content, height=200)
    key = st.number_input("Clé (nombre de rails)", min_value=2, step=1, value=2)

    action = st.radio("Action", ["Chiffrement", "Déchiffrement"])
    if st.button("Exécuter"):
        if action == "Chiffrement":
            st.success(f"Texte chiffré : {rail_fence_encrypt(text, key)}")
        else:
            st.success(f"Texte déchiffré : {rail_fence_decrypt(text, key)}")

elif menu == "Chiffrement Vigenère":
    st.write('<center><h1 style="color:#FF6666;">Chiffrement Vigenère</h1></center>', unsafe_allow_html=True)
    option = st.radio("Choisissez une action", ["Aucune sélection","Chiffrement/Déchiffrement", "Cryptanalyse"])
    if option== "Aucune sélection":
        st.write("Veuillez sélectionner une option.")
    elif option == "Chiffrement/Déchiffrement":
        # Section pour importer un fichier texte
        uploaded_file = st.file_uploader("Veuillez saisir ou importer un texte", type=["txt"])

        # Texte à chiffrer/déchiffrer
        if uploaded_file is not None:
            # Lecture du contenu du fichier texte
            file_content = uploaded_file.read().decode("utf-8")
            st.info("Le contenu du fichier a été pré-rempli. Vous pouvez le modifier.")
        else:
            # Si aucun fichier n'est importé, initialiser un contenu vide
            file_content = ""

        # Champ texte avec le contenu du fichier pré-rempli (ou vide par défaut)
        text = st.text_area("Texte à chiffrer/déchiffrer", file_content, height=200)

        key = st.text_input("Clé de chiffrement", "")
        action = st.radio("Action", ["Chiffrement", "Déchiffrement"])
        if st.button("Exécuter"):
            if key:
                if action == "Chiffrement":
                    st.success(f"Texte chiffré : {vigenere_encrypt(text, key)}")
                else:
                    st.success(f"Texte déchiffré : {vigenere_decrypt(text, key)}")
            else:
                st.error("Veuillez entrer une clé de chiffrement.")

    elif option == "Cryptanalyse":
        
        #Section pour importer un fichier texte
        uploaded_file = st.file_uploader("Veuillez saisir ou importer un texte", type=["txt"])

        # Texte à chiffrer/déchiffrer
        if uploaded_file is not None:
            # Lecture du contenu du fichier texte
            file_content = uploaded_file.read().decode("utf-8")
            st.info("Le contenu du fichier a été pré-rempli. Vous pouvez le modifier.")
        else:
            # Si aucun fichier n'est importé, initialiser un contenu vide
            file_content = ""

        # Champ texte avec le contenu du fichier pré-rempli (ou vide par défaut)
        cipher = st.text_area("Texte à chiffrer/déchiffrer", file_content, height=200)

        method = st.radio("Méthode de cryptanalyse", ["Babbage", "Friedman"])
        if st.button("Analyser"):
            if cipher:
                if method == "Babbage":
                    # Exemple d'utilisation avec le texte que vous avez fourni
                    min_block_size = st.number_input("Donner une taille minimum de block :", min_value=2, step=1, value=3)
                    # Calculer les distances entre les répétitions des blocs et leurs facteurs premiers
                    results, all_factors = calculate_distance_and_prime_factors(cipher, min_block_size)

                    # Trouver les facteurs premiers les plus fréquents triés par ordre décroissant
                    sorted_factors = most_frequent_prime_factors(all_factors)

                    # Afficher les résultats
                    st.subheader("Résultats des blocs et de leurs facteurs premiers:")
                    for block, positions, distance, factors in results:
                        st.write(f"Bloc: {block} | Positions: {positions} | Distance: {distance} | Facteurs premiers: {sorted(factors)}")

                    # Afficher les facteurs premiers les plus fréquents par ordre décroissant
                    st.subheader("\nLes facteurs premiers les plus fréquents par ordre décroissant de fréquence:")
                    for factor, count in sorted_factors:
                        st.write(f"Facteur premier: {factor} | Fréquence: {count}")
                        
                    length_key=sorted_factors[0][0]
                    st.markdown(f"<h3>La taille de la clé est : {length_key}</h3>", unsafe_allow_html=True)

                    #___
                    # Extraire les lettres par modulo k
                    extracted = extract_letters_by_modulo_k(cipher, length_key)

                    # Vecteur pour sauvegarder la lettre la plus fréquente
                    extracted_letters = " "
                    # Affichage des résultats et sauvegarde dans le vecteur
                    for p, letters in extracted.items():
                        # Calculer la lettre la plus fréquente
                        letter, freq = calculate_highest_letter_frequency(letters)
                        
                        if letter is not None:
                            # Ajouter la lettre la plus fréquente à la fin du vecteur (extracted_letters)
                            extracted_letters += letter
                            st.write(f"\nLettres aux positions P mod {length_key} = {p}:")
                            st.write("  Lettre la plus fréquente : ", letter)
                        else:
                            st.write(f"\nAucune lettre extraite pour P mod {length_key} = {p}")

                    # Afficher toutes les lettres les plus fréquentes dans le vecteur
                    st.markdown('<p style="color:#FF6666; text-align:center;">Lettres les plus fréquentes extraites (une par position modulaire) :</p>',unsafe_allow_html=True)

                    st.markdown(
                        f'<h2 style="text-align:center; color:white;">{extracted_letters}</h2>',
                        unsafe_allow_html=True
                    )

                    key=solve_multiple_equations(extracted_letters)
                    st.markdown('<p style="color:#FF6666; text-align:center;">La clé du chiffrement est :</p>',unsafe_allow_html=True)

                    st.markdown(
                        f'<h2 style="text-align:center; color:white;">{key}</h2>',
                        unsafe_allow_html=True
                    )
                
                elif method == "Friedman":
                    # Exemple d'utilisation :
                    m = st.number_input("Donnez une taille de clé max :", min_value=2, step=1, value=6)

                    # Extraire les séquences avec des pas allant de 1 à m et calculer leur IC
                    sequences_and_ic = extract_sequences_with_steps_and_calculate_ic(cipher, m)
                    st.subheader('Calcul des frequences:')
                    # Afficher les résultats pour chaque pas
                    for idx, (seq, ic) in enumerate(sequences_and_ic, start=1):
                        st.write(f"Pas {idx}: Séquence = {''.join(seq)}, IC = {ic:.4f}")

                    # Trouver la séquence avec l'IC le plus proche de 0,074 et le pas correspondant
                    sequence, ic, best_k = find_closest_ic_to_target(cipher, m, target_ic=0.074)

                    # Afficher le résultat
                    st.write(f"IC le plus proche de 0,074 : {ic:.4f}")
                    st.write(f"Séquence correspondante : {''.join(sequence)}")
                    st.write(f"Pas correspondant : {best_k}")
                    st.markdown(f"<h3>La taille de la clé est : {best_k}</h3>", unsafe_allow_html=True)
                    length_key=best_k   

                    # Extraire les lettres par modulo k
                    extracted = extract_letters_by_modulo_k(cipher, length_key)

                    # Vecteur pour sauvegarder la lettre la plus fréquente
                    extracted_letters = " "
                    # Affichage des résultats et sauvegarde dans le vecteur
                    for p, letters in extracted.items():
                        # Calculer la lettre la plus fréquente
                        letter, freq = calculate_highest_letter_frequency(letters)
                        
                        if letter is not None:
                            # Ajouter la lettre la plus fréquente à la fin du vecteur (extracted_letters)
                            extracted_letters += letter
                            st.write(f"\nLettres aux positions P mod {length_key} = {p}:")
                            st.write("  Lettre la plus fréquente : ", letter)
                        else:
                            st.write(f"\nAucune lettre extraite pour P mod {length_key} = {p}")
                    # Afficher toutes les lettres les plus fréquentes dans le vecteur
                    st.markdown('<p style="color:#FF6666; text-align:center;">Lettres les plus fréquentes extraites (une par position modulaire) :</p>',unsafe_allow_html=True)

                    st.markdown(
                        f'<h2 style="text-align:center; color:white;">{extracted_letters}</h2>',
                        unsafe_allow_html=True
                    )

                    key=solve_multiple_equations(extracted_letters)
                    st.markdown('<p style="color:#FF6666; text-align:center;">La clé du chiffrement est :</p>',unsafe_allow_html=True)

                    st.markdown(
                        f'<h2 style="text-align:center; color:white;">{key}</h2>',
                        unsafe_allow_html=True
                    )
                
    

            else:
                st.error("Veuillez entrer un texte chiffré.")
            
