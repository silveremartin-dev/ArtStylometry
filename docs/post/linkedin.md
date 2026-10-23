Et si l'on cessait de demander à l'IA « qui a peint ce tableau ? » ?
Pour lui poser une question plus honnête : « quelle zone de cette toile s'écarte statistiquement du corpus de référence ? »

C'est le point de départ d'ArtStylometry, un moteur open source de stylométrie computationnelle pour la peinture que je publie aujourd'hui.

🎨 Le problème
Les réseaux de vision classiques reconnaissent un sujet (une Vierge, un drapé) plutôt qu'une facture. Ils se laissent piéger par le scanner, l'éclairage ou les craquelures. Et ils ignorent la réalité des ateliers, où le maître et ses élèves se partagent une même toile.

🔬 L'approche
Inverser le problème : ne pas « prouver » une attribution, mais détecter des anomalies, patch par patch, à l'intérieur d'un corpus établi.

Ce que contient le dépôt :
• Collecte IIIF haute résolution (registre de 14 institutions : Rijksmuseum, Louvre, National Gallery, Met…)
• Découpage en patchs à l'échelle physique (mm), filtrage passe-bande contre les craquelures
• Embeddings DINOv2 / matrices de Gram, avec un fallback déterministe sans GPU
• Profils d'artistes (covariance Ledoit-Wolf / OAS, distance de Mahalanobis)
• Cartes de chaleur intra-toile et clustering HDBSCAN + UMAP
• 28 tests automatisés, licence MIT

⚠️ Statut, en toute honnêteté
C'est une première version, à un stade précoce. Le pipeline est implémenté et testé, mais le benchmark en 3 étapes tourne pour l'instant sur des vecteurs synthétiques. La validation sur de vraies œuvres (Rembrandt / Hals / Vermeer) est la prochaine étape. Et aucun score ne vaut certificat d'authenticité : sans IRR, radiographie X ni XRF, une image en lumière visible ne suffit pas.

🤝 Vos retours sont les bienvenus, en particulier ceux des historiens de l'art, des restaurateurs et des spécialistes de vision par ordinateur.
👉 https://github.com/silveremartin-dev/ArtStylometry

#HumanitésNumériques #HistoireDeLArt #ComputerVision #OpenSource #IIIF #MachineLearning
