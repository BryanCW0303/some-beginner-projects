import random

def draw(hand_card, cards):
    select_card = random.choice(cards)
    hand_card += [select_card]
    cards.remove(select_card)
    return hand_card

def result(card_1, card_2):
    print(f"Your cards: {card_1}")
    print(f"Computer's cards: {card_2}")
    point_1 = sum(card_1)
    point_2 = sum(card_2)
    if point_1 > 21:
        print("YOU LOSE!")
    elif point_1 < 21 and point_2 > 21:
        print("YOU WIN!")
    elif point_1 < point_2:
        print("YOU LOSE!")
    elif point_1 >= point_2:
        print("YOU WIN!")
    elif point_1 == 21 or point_2 == 21:
        print("BLACKJACK")
        if point_1 == 21:
            print("YOU WIN!")
        else:
            print("YOU LOSE")

def blackjack():
    cards = []
    for i in range(1,14):
        if i == 1:
            cards += [11, 11, 11, 11]
        elif i >= 2 and i <= 10:
            cards += [i, i, i, i]
        elif i > 10:
            cards += [10, 10, 10, 10]
    random.shuffle(cards)

    your_card = []
    computer_card = []
    your_card = draw(your_card, cards)
    computer_card = draw(computer_card, cards)
    your_card = draw(your_card, cards)
    computer_card = draw(computer_card, cards)

    print(f"Your cards: [{your_card}]")
    print(f"Computer's first card: {computer_card[0]}")

    another_card = input("Type 'y' to get another card, type 'n' to pass: ")
    while another_card == "y":
        your_card = draw(your_card, cards)
        print(f"Your cards: [{your_card}]")
        another_card = input("Type 'y' to get another card, type 'n' to pass: ")

    result(your_card, computer_card)

start = input("Do you want to play a game of Blackjack? Type 'y' or 'n': ")

while start == "y":
    blackjack()
    start = input("Do you want to play it again? Type 'y' or 'n': ")

print("Thank you for playing!")