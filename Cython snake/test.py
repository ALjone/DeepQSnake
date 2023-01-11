import curses
import numpy as np

import snake  # This is the Cython module you created
from game import snake_env
def main():
    # Initialize the game
    game = snake_env(10, 10, 100)
    #game = snake.SnakeGame(10, 10, 100)
    stdscr = curses.initscr()
    stdscr.addstr(str(game))
    # Don't display user input
    curses.noecho()
    # React to keys without pressing enter (700ms delay)
    curses.halfdelay(3)
    # Enumerate keys
    stdscr.keypad(True)

    # Game loop
    total_reward = 0
    while True:
        # Get user input
        action = -1
        while action == -1:
            action = stdscr.getch()
        print(action)
        # Convert ASCII code to action (0: up, 1: right, 2: down, 3: left)
        if action == 259:#curses.KEY_UP:
            action = 0
        elif action == 261:#curses.KEY_RIGHT:
            action = 1
        elif action == 258:#curses.KEY_DOWN:
            action = 2
        elif action == 260:#curses.KEY_LEFT:
            action = 3
        else:
            action = 3

        # Update the game state
        _, reward, done = game.step(action)
        total_reward += reward
        # Clear the screen
        stdscr.clear()

        # Print the game state
        stdscr.addstr(str(game))
        stdscr.addstr("\n\n" + "".join([str(x) for x in game.valid_moves()]))
        stdscr.addstr("\nLast reward: " + str(reward))
        stdscr.addstr("\nTotal reward: " + str(total_reward))

        # Check if the game is done
        if done:
            if reward == 1:
                stdscr.addstr("You won!")
            else:
                stdscr.addstr("Game over!")
            break

        # Refresh the screen
        stdscr.refresh()
        #print(np.array(game.get_state()))
    print("Last reward (should be -1):", reward)
    print("Total reward:", total_reward)

if __name__ == "__main__":
    # Run the main function with curses
    main()