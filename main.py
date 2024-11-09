import kagglehub
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plots(df):

    # Count the number of instances (rows) for each country
    country_counts = df["Country"].value_counts()

    # Create a bar plot for the number of instances per country
    plt.figure(figsize=(14, 8))
    sns.barplot(x=country_counts.index, y=country_counts.values, color="skyblue")

    # Title and labels
    plt.title("Number of Instances per Country")
    plt.xlabel("Country")
    plt.ylabel("Number of Instances")

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=90)

    # Show the plot
    plt.tight_layout()
    plt.show()

    # Plot missing data counts per column
    plt.figure(figsize=(10, 6))
    missing_data = df.isnull().sum()
    missing_data[missing_data > 0].plot(kind="bar", color="orange")

    # Title and labels
    plt.title("Number of Missing Values Per Column")
    plt.xlabel("Fields (Columns)")
    plt.ylabel("Number of Missing Values")

    # Show plot
    plt.xticks(rotation=45, ha="right")
    plt.show()

    # Plot life expectancy distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(df["Life expectancy"], kde=True, bins=30, color="skyblue")
    plt.title("Distribution of Life Expectancy")
    plt.xlabel("Life Expectancy")
    plt.ylabel("Frequency")
    plt.show()

    # Plot population distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(df["Population"], bins=10, color="green")
    plt.title("Distribution of Population")
    plt.xlabel("Population")
    plt.ylabel("Frequency")
    plt.show()

    # Plot income composition distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(
        df["Income composition of resources"], kde=True, bins=30, color="purple"
    )
    plt.title("Distribution of Income Composition of Resources")
    plt.xlabel("Income Composition of Resources")
    plt.ylabel("Frequency")
    plt.show()

    # Plot schooling distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(df["Schooling"], kde=True, bins=30, color="orange")
    plt.title("Distribution of Schooling (Years of Education)")
    plt.xlabel("Schooling (Years)")
    plt.ylabel("Frequency")
    plt.show()

    # Plot GDP vs Life Expectancy
    gdp_null_or_zero = df[df["GDP"].isna() | (df["GDP"] == 0)]
    print(gdp_null_or_zero)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x="GDP", y="Life expectancy", data=df, color="blue", alpha=0.6)
    plt.title("GDP vs Life Expectancy")
    plt.xlabel("GDP")
    plt.ylabel("Life Expectancy")
    plt.show()


def info(df):
    # 1. Number of instances (rows) in the dataset
    num_instances = df.shape[0]

    # 2. Number of different countries
    num_countries = df["Country"].nunique()
    un_countries = df["Country"].unique()

    # 3. Number of different years
    num_years = df["Year"].nunique()
    un_years = df["Year"].unique()

    # 4. Number of instances for each country
    country_counts = df["Country"].value_counts()

    # 5. Number of instances per year per country
    instances_per_year_country = (
        df.groupby(["Country", "Year"]).size().reset_index(name="Instances")
    )

    print(f"1. Total number of instances (rows) in the dataset: {num_instances}")

    print(f"2. Number of different countries: {num_countries}")
    print(un_countries)

    print(f"\n3. Number of different years: {num_years}")
    print(un_years)

    print(f"\n4. Number of instances for each country:")
    print(country_counts)

    print(f"\n5. Number of instances per year per country:")
    print(instances_per_year_country)


def main():
    # Download dataset
    path = kagglehub.dataset_download("kumarajarshi/life-expectancy-who")
    print("Path to dataset files:", path)
    data_path = f"{path}/Life Expectancy Data.csv"
    df = pd.read_csv(data_path)
    df.columns = df.columns.str.strip()

    info(df)

    plots(df)


if __name__ == "__main__":
    main()
