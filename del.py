import msvcrt


user_input = b''

while True:
    pressed_key = msvcrt.getche()  # getch() will not echo key to window if that is what you want
    if pressed_key == b'\x1b':  # b'\x1b' is escape
        raise SystemExit
    elif pressed_key == b'\r':  # b'\r' is enter or return
        break
    else:
        user_input += pressed_key

print('\n' + user_input.decode())  # this just shows you that user_input variable can be used now somewhere else in your code
input()  # input just 