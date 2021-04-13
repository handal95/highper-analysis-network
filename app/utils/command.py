import structlog
from app.exc import QuitException 

logger = structlog.get_logger()


def request_user_input(
    msg="Please enter any key to proceed to the next step.",
    valid_inputs=None, valid_outputs=None, default=None, skipable=False
):
    if valid_inputs is None:
        return input(msg)

    skip_msg = None
    assert 'Q' not in valid_inputs
    if skipable:
        assert 'S' not in valid_inputs
        skip_msg = ", 'S' for the skip the progress"

    valid_inputs = [str(x).capitalize() for x in valid_inputs]

    if msg:
        logger.info(msg)

    if valid_outputs is None:
        valid_outputs = valid_inputs

    assert len(valid_inputs) == len(valid_outputs)

    while True:
        user_input = input() or default

        if user_input:
            user_input = user_input.capitalize()
            
            if user_input == 'Q':
                raise QuitException
            
            if skipable and user_input == 'S':
                return None

            if user_input not in valid_inputs:
                logger.info(f"'{user_input}' is not a valid input !")
                logger.info(
                    f"Answer must in the valid input list{valid_inputs}  "
                    f"('Q' for the quit the progress{skip_msg})"
                )
            else:
                return valid_outputs[valid_inputs.index(user_input)]
                
        elif default:
            user_input if user_input else default
        