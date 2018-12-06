import clean_data
import add_result
clean_data.main(['','train'])
clean_data.main(['','validate'])
clean_data.main(['','test'])
add_result.main(['','train'])
add_result.main(['','validate'])
add_result.main(['','test'])