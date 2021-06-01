from selenium import webdriver
driver=webdriver.Chrome("/lib/chromium-browser/chromedriver")
driver.get("http://thedemosite.co.uk/login.php")
driver.find_element_by_name("username").send_keys("test")
driver.find_element_by_name("password").send_keys("test")
driver.find_element_by_xpath("/html/body/table/tbody/tr/td[1]/form/div/center/table/tbody/tr/td[1]/table/tbody/tr[3]/td[2]/p/input").click()
