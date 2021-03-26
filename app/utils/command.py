import structlog
from app.exc import QuitException 

logger = structlog.get_logger()


def request_user_input(msg, valid_inputs=None, default=None):
    assert 'Q' not in valid_inputs

    if msg:
        logger.info(msg)


    while True:
        user_input = input()

        if user_input:
            user_input = user_input.capitalize()
            
            if user_input == 'Q':
                raise QuitException

            if user_input not in valid_inputs:
                logger.info(
                    f"'{user_input}' is not a valid input !"
                    f"Answer must in the valid input list{valid_inputs}  "
                    f"('Q' for the quit the progress)"
                )
            else:
                return user_input

        elif default:
            user_input if user_input else default
        

