from app.run import run
from app.utils.logger import init_logger



if __name__ == '__main__':
    try:
        init_logger()
        run()
    except KeyboardInterrupt:
        logger.warn(f'Abort!')
