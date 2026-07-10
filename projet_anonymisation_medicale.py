import cv2
import os
import argparse
import time
import matplotlib.pyplot as plt


FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)


def verifier_detecteur():
    """Vérifie que le fichier de détection est bien chargé."""
    if FACE_CASCADE.empty():
        raise RuntimeError("Erreur : le détecteur de visage OpenCV n'a pas pu être chargé.")


def creer_dossier_si_absent(dossier):
    """Crée un dossier s'il n'existe pas."""
    if dossier and not os.path.exists(dossier):
        os.makedirs(dossier)


def detecter_visages(image):
    """Retourne la liste des visages détectés dans une image."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(40, 40)
    )
    return faces


def anonymiser_zone(image, x, y, w, h, mode='flou'):
    """Anonymise une zone du visage par flou ou masque noir."""
    roi = image[y:y + h, x:x + w]

    if mode == 'flou':
        zone_anonymisee = cv2.GaussianBlur(roi, (99, 99), 30)
        image[y:y + h, x:x + w] = zone_anonymisee
    elif mode == 'masque':
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 0), -1)
    else:
        raise ValueError("Mode invalide. Utilisez 'flou' ou 'masque'.")

    return image


def anonymiser_image(image_path, output_path, mode='flou'):
    """Détecte les visages dans une image puis les anonymise."""
    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(f"Impossible de lire l'image : {image_path}")

    debut = time.time()
    faces = detecter_visages(image)
    image_resultat = image.copy()

    for (x, y, w, h) in faces:
        anonymiser_zone(image_resultat, x, y, w, h, mode=mode)

    creer_dossier_si_absent(os.path.dirname(output_path))
    cv2.imwrite(output_path, image_resultat)
    duree = time.time() - debut

    return {
        'type': 'image',
        'input': image_path,
        'output': output_path,
        'visages_detectes': len(faces),
        'mode': mode,
        'duree_secondes': round(duree, 3)
    }


def verifier_anonymisation(image_path):
    """Vérifie si un visage est encore détecté après anonymisation."""
    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(f"Impossible de lire l'image : {image_path}")

    faces_restants = detecter_visages(image)

    return {
        'image_verifiee': image_path,
        'visages_restants': len(faces_restants),
        'anonymisation_reussie': len(faces_restants) == 0
    }


def comparer_avant_apres(image_originale, image_anonymisee, chemin_figure=None):
    """Affiche et/ou sauvegarde une comparaison avant/après."""
    img1 = cv2.imread(image_originale)
    img2 = cv2.imread(image_anonymisee)

    if img1 is None or img2 is None:
        raise FileNotFoundError("Impossible de lire une des images pour la comparaison.")

    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(img1)
    plt.title('Avant anonymisation')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(img2)
    plt.title('Après anonymisation')
    plt.axis('off')

    plt.tight_layout()

    if chemin_figure:
        creer_dossier_si_absent(os.path.dirname(chemin_figure))
        plt.savefig(chemin_figure, bbox_inches='tight')

    plt.show()


def anonymiser_video(video_path, output_path, mode='flou'):
    """Détecte les visages dans une vidéo puis les anonymise image par image."""
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise FileNotFoundError(f"Impossible d'ouvrir la vidéo : {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25

    creer_dossier_si_absent(os.path.dirname(output_path))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    debut = time.time()
    total_frames = 0
    total_visages = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = detecter_visages(frame)
        total_visages += len(faces)

        for (x, y, w, h) in faces:
            anonymiser_zone(frame, x, y, w, h, mode=mode)

        out.write(frame)
        total_frames += 1

    cap.release()
    out.release()
    duree = time.time() - debut

    return {
        'type': 'video',
        'input': video_path,
        'output': output_path,
        'frames_traitees': total_frames,
        'visages_detectes_total': total_visages,
        'mode': mode,
        'duree_secondes': round(duree, 3)
    }


def main():
    verifier_detecteur()

    parser = argparse.ArgumentParser(
        description="Projet de sécurité des données médicales : anonymisation d'images et vidéos."
    )
    parser.add_argument('--type', choices=['image', 'video'], required=True, help="Type d'entrée à traiter")
    parser.add_argument('--input', required=True, help="Chemin de l'image ou de la vidéo d'entrée")
    parser.add_argument('--output', required=True, help="Chemin du fichier de sortie anonymisé")
    parser.add_argument('--mode', choices=['flou', 'masque'], default='flou', help="Méthode d'anonymisation")
    parser.add_argument('--verifier', action='store_true', help="Vérifie l'anonymisation après traitement (image uniquement)")
    parser.add_argument('--comparer', action='store_true', help="Affiche la comparaison avant/après (image uniquement)")
    parser.add_argument('--figure', default=None, help="Chemin de sauvegarde de la figure de comparaison")
    args = parser.parse_args()

    if args.type == 'image':
        resultat = anonymiser_image(args.input, args.output, mode=args.mode)
        print("=== RÉSULTAT IMAGE ===")
        print(resultat)

        if args.verifier:
            verification = verifier_anonymisation(args.output)
            print("=== VÉRIFICATION ===")
            print(verification)

        if args.comparer:
            comparer_avant_apres(args.input, args.output, chemin_figure=args.figure)

    elif args.type == 'video':
        resultat = anonymiser_video(args.input, args.output, mode=args.mode)
        print("=== RÉSULTAT VIDÉO ===")
        print(resultat)
        print("La comparaison visuelle se fait en lisant la vidéo originale puis la vidéo anonymisée.")


if __name__ == '__main__':
    main()
