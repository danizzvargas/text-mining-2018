import clean_data
import add_result
clean_data.main(['','train'])
clean_data.main(['','validate'])
add_result.main(['','train'])
add_result.main(['','validate'])