#!/bin/bash
# Script pour lancer tous les tests de qualité de code
# Usage: ./run_quality_tests.sh

set -e  # Arrêter en cas d'erreur

# Couleurs pour l'output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Répertoire du projet
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

# Activer l'environnement virtuel si disponible
if [ -d "SP_Forecasting/bin/activate" ]; then
    source SP_Forecasting/bin/activate
fi

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Tests de Qualité de Code${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Compteur d'erreurs
ERRORS=0

# 1. Ruff - Linting et formatage
echo -e "${YELLOW}[1/6] Ruff - Linting et formatage...${NC}"
if ruff check src/ --output-format=concise; then
    echo -e "${GREEN}✓ Ruff: OK${NC}"
else
    echo -e "${RED}✗ Ruff: Erreurs détectées${NC}"
    ERRORS=$((ERRORS + 1))
fi
echo ""

# 2. Black - Formatage (vérification)
echo -e "${YELLOW}[2/6] Black - Vérification du formatage...${NC}"
if black --check src/ --line-length 100; then
    echo -e "${GREEN}✓ Black: OK${NC}"
else
    echo -e "${RED}✗ Black: Code non formaté${NC}"
    ERRORS=$((ERRORS + 1))
fi
echo ""

# 3. Bandit - Sécurité
echo -e "${YELLOW}[3/6] Bandit - Analyse de sécurité...${NC}"
if bandit -r src/ -f txt -ll; then
    echo -e "${GREEN}✓ Bandit: OK${NC}"
else
    echo -e "${RED}✗ Bandit: Problèmes de sécurité détectés${NC}"
    ERRORS=$((ERRORS + 1))
fi
echo ""

# 4. Radon - Complexité cyclomatique
echo -e "${YELLOW}[4/6] Radon - Complexité cyclomatique...${NC}"
if radon cc src/ --min B --show-complexity > /tmp/radon_cc.txt 2>&1; then
    COMPLEX_FUNCTIONS=$(grep " - [CDEF] (" /tmp/radon_cc.txt | wc -l | tr -d ' ' || echo 0)
    if [ "${COMPLEX_FUNCTIONS:-0}" -gt 0 ]; then
        echo -e "${RED}✗ Radon: $COMPLEX_FUNCTIONS fonctions avec complexité supérieure à B (C, D, E ou F)${NC}"
        echo "Fonctions concernées:"
        grep " - [CDEF] (" /tmp/radon_cc.txt | head -10
        ERRORS=$((ERRORS + 1))
    else
        echo -e "${GREEN}✓ Radon: Complexité acceptable (B ou mieux)${NC}"
    fi
else
    echo -e "${RED}✗ Radon: Erreur lors de l'analyse${NC}"
    ERRORS=$((ERRORS + 1))
fi
echo ""

# 5. Radon - Maintenance Index
echo -e "${YELLOW}[5/6] Radon - Index de maintenance...${NC}"
radon mi src/ --min B > /tmp/radon_mi.txt 2>&1 || true
LOW_MAINTENANCE=$(grep " - [CDEF] " /tmp/radon_mi.txt 2>/dev/null | wc -l | tr -d ' ' || echo 0)
if [ "${LOW_MAINTENANCE:-0}" -gt 0 ]; then
    echo -e "${RED}✗ Radon MI: $LOW_MAINTENANCE fichiers avec index de maintenance inférieur à B (C, D, E ou F)${NC}"
    ERRORS=$((ERRORS + 1))
else
    echo -e "${GREEN}✓ Radon MI: Index de maintenance acceptable (B ou mieux)${NC}"
fi
echo ""

# 6. Xenon - Surveillance de complexité
echo -e "${YELLOW}[6/6] Xenon - Surveillance de complexité...${NC}"
xenon --max-absolute B --max-modules B --max-average B src/ > /tmp/xenon.txt 2>&1 || true
COMPLEX_BLOCKS=$(grep "ERROR:xenon" /tmp/xenon.txt 2>/dev/null | wc -l | tr -d ' ' || echo 0)
if [ "${COMPLEX_BLOCKS:-0}" -gt 0 ]; then
    echo -e "${YELLOW}⚠ Xenon: $COMPLEX_BLOCKS blocs avec complexité élevée${NC}"
    grep "ERROR:xenon" /tmp/xenon.txt | head -10
else
    echo -e "${GREEN}✓ Xenon: Complexité acceptable${NC}"
fi
echo ""

# Résumé
echo -e "${GREEN}========================================${NC}"
if [ $ERRORS -eq 0 ]; then
    echo -e "${GREEN}Tous les tests sont passés !${NC}"
    exit 0
else
    echo -e "${RED}$ERRORS test(s) ont échoué${NC}"
    exit 1
fi

