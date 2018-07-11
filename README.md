# Portimize

A portfolio is a grouping of financial assets such as stocks, bonds, commodities, currencies and cash equivalents, as well as their fund counterparts, including mutual, exchange-traded and closed funds. Portfolios are held directly by investors and/or managed by financial professionals and money managers. Portimize gives the user an option to hold a portfolio either on a short-term basis or for long-term holdings. 

Investors should construct an investment portfolio in accordance with their risk tolerance and their investing objectives. This project involves implementing the Markowitz Portolio Optimization in order to maximize the returns at a minimum risk given historical prices. This would be the case for the longer holdings whereas for the short term trading purposes, portimize uses a Long short-term memory network trained for different scenarios independently to forecast the OHLC prices of a security. Moreover, these prices are then statistically used to provide an optimal allocation for your chosen securities within the stipulated time.


## Getting Started

### Prerequisites

Apart from Python 2.7, there are certain dependencies this project has:

```
Django (1.11.1)
numpy (1.12.1)
pandas (0.20.1)
pandas-datareader (0.5.0)
scipy (0.19.0)
Keras (2.1.3)
tensorflow (1.4.1)
plotly (2.1.0)
h5py (2.7.1)
```

### Setup
* Clone this repository to your local machine.
* Install all the dependencies (preferably in a virtualenv).
* Run `cd Portimize` and then `python manage.py runserver`
* Now open `localhost:8000` in your browser.


## Contributing

### Contribution Guidelines
* Pull requests to be sent from any branch of the fork except `master`
* Commit messages should be descriptive. For example `feat:Added templates for Prediction Pages`

### Pull Request Creation Help
* `git add *`
* `git commit -m "Descriptive Commit Message`
* `git push -u origin branchname`

### Fork Master Synchronization
* Assuming main repo has been added as a remote named `upstream`
* `git fetch upstream`
* `git reset --hard upstream/master`
 
### Fork Branch updation with master
* Assuming `feature-branch` is checked out.
* Run `git rebase master` to update your branch if it is behind `master`(local)
* If conflicts arise fix them and run `git add *` and then `git rebase --continue`

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details



