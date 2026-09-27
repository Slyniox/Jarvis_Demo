from parser import DiaryParser
from database import DiaryDatabase


def main():

    parser = DiaryParser("LJDB.md")

    entries = parser.parse()

    db = DiaryDatabase("LJDB_FR.db")

    db.insert_entries(entries)

    print(
        f"Database now contains "
        f"{db.number_of_entries()} entries."
    )

    db.close()


if __name__ == "__main__":
    main()